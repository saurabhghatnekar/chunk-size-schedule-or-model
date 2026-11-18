"""Gated linear attention (Yang et al., arXiv 2312.06635), chunkwise form.

Recurrence (per head, state S in R^{K x V}, vector gate alpha_t in (0,1)^K):

    S_t = diag(alpha_t) S_{t-1} + k_t^T v_t
    o_t = q_t S_t

Chunkwise form for a chunk that starts from state S_0, with the within-chunk
inclusive cumulative log-gate b_t = sum_{r<=t} log alpha_r:

    o_t   = (q_t * exp(b_t)) S_0 + sum_{i<=t} [ sum_d q_td k_id exp(b_td - b_id) ] v_i
    S_end = diag(exp(b_L)) S_0   + sum_i (k_i * exp(b_L - b_i))^T v_i

The pairwise decay exp(b_t - b_i) is evaluated in log space, so nothing is ever
divided by a vanishing cumulative product. That costs O(L^2 K) memory per chunk,
which is fine for a reference and is the reason production kernels use a
secondary level of chunking instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, chunk_bounds, merge_heads, split_heads


def recurrent_gla(q, k, v, log_alpha):
    """Token-by-token recurrence. q,k,log_alpha: (B,H,T,K); v: (B,H,T,V)."""
    b, h, t, kd = q.shape
    state = q.new_zeros(b, h, kd, v.shape[-1])
    outs = []
    for i in range(t):
        state = state * log_alpha[:, :, i].exp()[..., None] + k[:, :, i, :, None] * v[:, :, i, None, :]
        outs.append((q[:, :, i, None, :] @ state).squeeze(-2))
    return torch.stack(outs, dim=2)


def chunk_gla(q, k, v, log_alpha, chunk_size: int):
    b, h, t, kd = q.shape
    state = q.new_zeros(b, h, kd, v.shape[-1])
    outs = []
    for s, e in chunk_bounds(t, chunk_size):
        qc, kc, vc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
        cum = log_alpha[:, :, s:e].cumsum(dim=2)  # (B,H,L,K), inclusive
        length = e - s
        inter = (qc * cum.exp()) @ state
        # pairwise log-decay b_t - b_i, only needed for i <= t
        diff = cum[:, :, :, None, :] - cum[:, :, None, :, :]  # (B,H,L,L,K)
        lower = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        diff = diff.masked_fill(~lower[None, None, :, :, None], 0.0)
        attn = torch.einsum("bhtk,bhik,bhtik->bhti", qc, kc, diff.exp())
        attn = attn.masked_fill(~lower, 0.0)
        outs.append(inter + attn @ vc)
        last = cum[:, :, -1:, :]  # (B,H,1,K)
        state = state * last.squeeze(2).exp()[..., None] + (kc * (last - cum).exp()).transpose(-1, -2) @ vc
    return torch.cat(outs, dim=2)


class GLAMixer(nn.Module):
    name = "gla"

    def __init__(self, d_model: int, n_heads: int, gate_rank: int = 16, tau: float = 16.0):
        super().__init__()
        self.n_heads = n_heads
        self.tau = tau
        d_key = d_model // 2
        self.q_proj = nn.Linear(d_model, d_key, bias=False)
        self.k_proj = nn.Linear(d_model, d_key, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gk_proj = nn.Sequential(nn.Linear(d_model, gate_rank, bias=False), nn.Linear(gate_rank, d_key))
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model // n_heads)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        q = split_heads(self.q_proj(x), self.n_heads)
        k = split_heads(self.k_proj(x), self.n_heads)
        v = split_heads(self.v_proj(x), self.n_heads)
        log_alpha = split_heads(F.logsigmoid(self.gk_proj(x)) / self.tau, self.n_heads)
        q = q * (q.shape[-1] ** -0.5)
        o = chunk_gla(q, k, v, log_alpha, chunk_size)
        o = merge_heads(self.norm(o))
        return self.o_proj(o * F.silu(self.g_proj(x)))
