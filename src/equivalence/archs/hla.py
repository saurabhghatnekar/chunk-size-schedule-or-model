"""HLA-style chunk-level routing on top of Gated DeltaNet (Chen et al., arXiv 2610.05842).

Token recurrence (same as `delta.py`): S_t = A_t S_{t-1} + B_t with
A_t = exp(g_t)(I - beta_t k_t^T k_t) and B_t = beta_t k_t^T v_t. A completed
chunk j of length L composes its token transitions into one affine map
(A_j, B_j). For a query token i in chunk c(i):

    pooled key of window b of chunk j:
        kbar_{j,b} = mean_rows[ softmax(K K^T / sqrt(d)) V ] W_o,  K = X_{j,b} W_k, V = X_{j,b} W_v
    routing weight (independent sigmoids, no sum-to-one constraint):
        s_{i,j} = log (1/P) sum_b exp( q^r_i . kbar_{j,b} / sqrt(d) ),   w_{i,j} = sigmoid(s_{i,j}),  j < c(i)
    query-conditioned history, applied over completed chunks in order from S = 0:
        S <- [(1 - w_{i,j}) I + w_{i,j} A_j] S + w_{i,j} B_j
    readout through the visible prefix of the current chunk (weight one):
        S_i = A_{c(i),<=i} S + B_{c(i),<=i},   o_i = q_i S_i

Setting every w to one recovers Gated DeltaNet exactly, for any chunk length
(`force_unit_routing=True`, used by the theory tests). With learned routing the
router acts on chunks, so moving the boundary changes what can be routed, and a
single chunk spanning the sequence has no history to route and is plain GDN.

This is written from the method section of the paper, not from released code;
treat it as HLA-style until checked against an official implementation. The
inference-time threshold on w and the training regulariser are not part of the
forward function tested here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, chunk_bounds, merge_heads
from .delta import DeltaMixer


def _chunk_transitions(k, v, beta, g, chunk_size: int):
    """Affine prefix maps inside each chunk.

    Returns (a_pref, b_pref): (B,H,T,K,K) and (B,H,T,K,V), where position t holds
    the composition of the token transitions from its chunk's first token up to
    and including t. The entry at a chunk's last token is that chunk's (A_j, B_j).
    """
    bsz, h, t, kd = k.shape
    eye = torch.eye(kd, dtype=k.dtype, device=k.device).expand(bsz, h, kd, kd)
    a_list, b_list = [], []
    a_cur = b_cur = None
    for i in range(t):
        if i % chunk_size == 0:
            a_cur = eye
            b_cur = k.new_zeros(bsz, h, kd, v.shape[-1])
        ki = k[:, :, i, None, :]  # (B,H,1,K)
        bi = beta[:, :, i, None, None]
        decay = g[:, :, i].exp()[..., None, None]
        a_cur = decay * (a_cur - bi * ki.transpose(-1, -2) @ (ki @ a_cur))
        b_cur = decay * (b_cur - bi * ki.transpose(-1, -2) @ (ki @ b_cur)) + bi * ki.transpose(-1, -2) * v[:, :, i, None, :]
        a_list.append(a_cur)
        b_list.append(b_cur)
    return torch.stack(a_list, dim=2), torch.stack(b_list, dim=2)


class HLAMixer(DeltaMixer):
    name = "hla"

    def __init__(self, d_model: int, n_heads: int, pool_window: int = 16, force_unit_routing: bool = False):
        super().__init__(d_model, n_heads, gated=True)
        self.pool_window = pool_window
        self.force_unit_routing = force_unit_routing
        self.route_q = nn.Linear(d_model, d_model, bias=False)
        self.pool_k = nn.Linear(d_model, d_model, bias=False)
        self.pool_v = nn.Linear(d_model, d_model, bias=False)
        self.pool_o = nn.Linear(d_model, d_model, bias=False)

    def _pooled_keys(self, x, bounds):
        """One pooled routing key per window of every completed chunk.

        Returns a list over chunks of tensors (B,H,P_j,D)."""
        bsz, _, d_model = x.shape
        h = self.n_heads
        pooled = []
        for s, e in bounds:
            reps = []
            for ws in range(s, e, self.pool_window):
                xw = x[:, ws:min(ws + self.pool_window, e)]
                kp, vp = self.pool_k(xw), self.pool_v(xw)
                attn = F.softmax(kp @ kp.transpose(-1, -2) / math.sqrt(d_model), dim=-1)
                reps.append(self.pool_o((attn @ vp).mean(dim=1)))  # (B,D)
            rep = torch.stack(reps, dim=1)  # (B,P,D)
            pooled.append(rep.view(bsz, -1, h, d_model // h).transpose(1, 2))
        return pooled

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        h = self.n_heads
        q, k, v = self.qkv(x)
        beta, g = self.gates(x)
        a_pref, b_pref = _chunk_transitions(k, v, beta, g, chunk_size)
        bounds = list(chunk_bounds(seq_len, chunk_size))
        pooled = self._pooled_keys(x, bounds[:-1])
        q_route = self.route_q(x).view(bsz, seq_len, h, d_model // h).transpose(1, 2)  # (B,H,T,D)
        head_dim = d_model // h

        outs = []
        for c, (s, e) in enumerate(bounds):
            length = e - s
            hist = q.new_zeros(bsz, h, length, k.shape[-1], v.shape[-1])
            for j in range(c):
                end_j = bounds[j][1] - 1
                a_j = a_pref[:, :, end_j, None]  # (B,H,1,K,K)
                b_j = b_pref[:, :, end_j, None]  # (B,H,1,K,V)
                if self.force_unit_routing:
                    hist = a_j @ hist + b_j
                    continue
                scores = q_route[:, :, s:e] @ pooled[j].transpose(-1, -2) / math.sqrt(head_dim)  # (B,H,L,P)
                w = torch.sigmoid(torch.logsumexp(scores, dim=-1) - math.log(scores.shape[-1]))
                w = w[..., None, None]
                hist = (1.0 - w) * hist + w * (a_j @ hist + b_j)
            state = a_pref[:, :, s:e] @ hist + b_pref[:, :, s:e]  # (B,H,L,K,V)
            outs.append((q[:, :, s:e, None, :] @ state).squeeze(-2))
        o = torch.cat(outs, dim=2)
        o = merge_heads(self.norm(o))
        return self.o_proj(o * F.silu(self.g_proj(x)))
