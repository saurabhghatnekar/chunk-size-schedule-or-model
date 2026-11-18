"""DeltaNet (arXiv 2406.06484) and Gated DeltaNet (arXiv 2412.06464), chunkwise form.

Recurrence (per head, state S in R^{K x V}, row-vector convention as in the
flash-linear-attention library; g_t <= 0 is the log of the scalar gate, and
g_t = 0 recovers plain DeltaNet):

    S_t = exp(g_t) (I - beta_t k_t^T k_t) S_{t-1} + beta_t k_t^T v_t
    o_t = q_t S_t

Chunkwise form. Inside one chunk starting from S_0, write gamma_t for the
inclusive cumulative sum of g and define the pseudo-value

    u_t = beta_t (v_t - k_t exp(g_t) S_{t-1}).

Unrolling S_{t-1} gives a unit lower-triangular system in the u's:

    (I + L) U = diag(beta) V - diag(beta exp(gamma)) K S_0,
    L[t, i]   = beta_t exp(gamma_t - gamma_i) (k_t . k_i)   for i < t,

and then

    o_t   = exp(gamma_t) q_t S_0 + sum_{i<=t} exp(gamma_t - gamma_i) (q_t . k_i) u_i
    S_end = exp(gamma_L) S_0     + sum_i      exp(gamma_L - gamma_i) k_i^T u_i.

Nothing here is an approximation: it is the same recurrence with the sums
re-associated. The triangular system is solved by plain forward substitution so
the code runs in every floating-point format, including the 16-bit ones.

`perturb` injects a deliberate defect and exists only for the sensitivity
control in `controls.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, chunk_bounds, l2norm, merge_heads, split_heads


def recurrent_gated_delta(q, k, v, beta, g):
    """Token-by-token recurrence. q,k: (B,H,T,K); v: (B,H,T,V); beta,g: (B,H,T)."""
    b, h, t, kd = q.shape
    state = q.new_zeros(b, h, kd, v.shape[-1])
    outs = []
    for i in range(t):
        ki = k[:, :, i, None, :]  # (B,H,1,K)
        state = state * g[:, :, i].exp()[..., None, None]
        u = beta[:, :, i, None, None] * (v[:, :, i, None, :] - ki @ state)  # (B,H,1,V)
        state = state + ki.transpose(-1, -2) @ u
        outs.append((q[:, :, i, None, :] @ state).squeeze(-2))
    return torch.stack(outs, dim=2)


def chunk_gated_delta(q, k, v, beta, g, chunk_size: int, perturb: dict | None = None):
    b, h, t, kd = q.shape
    state = q.new_zeros(b, h, kd, v.shape[-1])
    outs = []
    for s, e in chunk_bounds(t, chunk_size):
        qc, kc, vc, bc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], beta[:, :, s:e]
        gam = g[:, :, s:e].cumsum(dim=-1)  # (B,H,L)
        length = e - s
        # exp(gamma_t - gamma_i) on and below the diagonal, zero above
        decay = (gam[..., :, None] - gam[..., None, :]).tril().exp().tril()
        lmat = (bc[..., :, None] * (kc @ kc.transpose(-1, -2)) * decay).tril(-1)
        rhs = bc[..., None] * vc - ((bc * gam.exp())[..., None] * kc) @ state
        rows = [rhs[..., 0, :]]
        for i in range(1, length):
            prev = torch.stack(rows, dim=-2)  # (B,H,i,V)
            rows.append(rhs[..., i, :] - (lmat[..., i, :i, None] * prev).sum(dim=-2))
        u = torch.stack(rows, dim=-2)  # (B,H,L,V)
        qk = (qc @ kc.transpose(-1, -2)) * decay
        outs.append((qc * gam.exp()[..., None]) @ state + qk @ u)

        carry_log = gam[..., -1]
        if perturb is not None and perturb["kind"] == "stale_gate":
            # off-by-one: the carried state misses the last token's gate
            carry_log = carry_log - g[:, :, e - 1]
        state = state * carry_log.exp()[..., None, None] + (
            kc * (gam[..., -1:] - gam).exp()[..., None]
        ).transpose(-1, -2) @ u
        if perturb is not None and perturb["kind"] == "boundary_leak":
            state = state * (1.0 + perturb["delta"])
    return torch.cat(outs, dim=2)


class DeltaMixer(nn.Module):
    """DeltaNet when `gated=False`, Gated DeltaNet when `gated=True`."""

    def __init__(self, d_model: int, n_heads: int, gated: bool = True, perturb: dict | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.gated = gated
        self.perturb = perturb
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        if gated:
            self.a_proj = nn.Linear(d_model, n_heads, bias=False)
            self.a_log = nn.Parameter(torch.log(torch.linspace(1.0, 4.0, n_heads)))
            self.dt_bias = nn.Parameter(torch.linspace(-4.0, -1.0, n_heads))
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model // n_heads)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def gates(self, x):
        beta = self.b_proj(x).sigmoid().transpose(1, 2)  # (B,H,T)
        if self.gated:
            g = (-self.a_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias)).transpose(1, 2)
        else:
            g = torch.zeros_like(beta)
        return beta, g

    def qkv(self, x):
        q = l2norm(split_heads(F.silu(self.q_proj(x)), self.n_heads))
        k = l2norm(split_heads(F.silu(self.k_proj(x)), self.n_heads))
        v = split_heads(F.silu(self.v_proj(x)), self.n_heads)
        return q, k, v

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        q, k, v = self.qkv(x)
        beta, g = self.gates(x)
        o = chunk_gated_delta(q, k, v, beta, g, chunk_size, perturb=self.perturb)
        o = merge_heads(self.norm(o))
        return self.o_proj(o * F.silu(self.g_proj(x)))
