"""Mamba-2 state space layer, chunked SSD algorithm (Dao and Gu, arXiv 2405.21060).

Per head, with scalar log-decay a_t <= 0, input x_t in R^P, B_t, C_t in R^N and
state h in R^{P x N}:

    h_t = exp(a_t) h_{t-1} + x_t B_t^T
    y_t = h_t C_t

The chunked algorithm is the four-step block decomposition of the same
lower-triangular operator (the `ssd_minimal` listing of the paper): diagonal
blocks, per-chunk end states, a recurrence over chunk states, and the
low-rank off-diagonal blocks. A ragged tail is zero padded, which is exact
because padded steps have a = 0 (decay one) and x = 0 (no write).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm


def recurrent_ssd(x, a, bmat, cmat):
    """x: (B,T,H,P); a: (B,T,H); bmat,cmat: (B,T,H,N)."""
    b, t, h, p = x.shape
    state = x.new_zeros(b, h, p, bmat.shape[-1])
    outs = []
    for i in range(t):
        state = state * a[:, i].exp()[..., None, None] + x[:, i, :, :, None] * bmat[:, i, :, None, :]
        outs.append((state @ cmat[:, i, :, :, None]).squeeze(-1))
    return torch.stack(outs, dim=1)


def _segsum(x):
    """x: (..., L) -> (..., L, L) with out[t, i] = sum_{r=i+1..t} x_r for i <= t, -inf above."""
    length = x.shape[-1]
    x = x[..., None].expand(*x.shape, length)
    strictly_lower = torch.ones(length, length, dtype=torch.bool, device=x.device).tril(-1)
    x = x.masked_fill(~strictly_lower, 0.0)
    out = x.cumsum(dim=-2)
    lower = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
    return out.masked_fill(~lower, float("-inf"))


def chunk_ssd(x, a, bmat, cmat, chunk_size: int):
    seq_len = x.shape[1]
    pad = (-seq_len) % chunk_size
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        bmat = F.pad(bmat, (0, 0, 0, 0, 0, pad))
        cmat = F.pad(cmat, (0, 0, 0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad))
    bsz, total, h, p = x.shape
    n_chunks = total // chunk_size
    x = x.view(bsz, n_chunks, chunk_size, h, p)
    bmat = bmat.view(bsz, n_chunks, chunk_size, h, -1)
    cmat = cmat.view(bsz, n_chunks, chunk_size, h, -1)
    a = a.view(bsz, n_chunks, chunk_size, h).permute(0, 3, 1, 2)  # (B,H,C,L)
    a_cum = a.cumsum(dim=-1)

    # 1. diagonal blocks
    decay_intra = _segsum(a).exp()
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", cmat, bmat, decay_intra, x)
    # 2. state at the end of each chunk, assuming a zero incoming state
    decay_states = (a_cum[..., -1:] - a_cum).exp()
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", bmat, decay_states, x)
    # 3. recurrence over chunk states
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = _segsum(F.pad(a_cum[..., -1], (1, 0))).exp()
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    # 4. contribution of the incoming state to each position
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", cmat, states, a_cum.exp())

    y = (y_diag + y_off).reshape(bsz, total, h, p)
    return y[:, :seq_len]


class Mamba2Mixer(nn.Module):
    name = "mamba2"

    def __init__(self, d_model: int, n_heads: int, d_state: int = 32, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_inner = expand * d_model
        self.n_heads = n_heads * expand
        self.head_dim = self.d_inner // self.n_heads
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.n_heads, bias=False)
        conv_dim = self.d_inner + 2 * d_state
        self.conv = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        self.a_log = nn.Parameter(torch.log(torch.linspace(1.0, 8.0, self.n_heads)))
        self.dt_bias = nn.Parameter(torch.linspace(-4.0, -1.0, self.n_heads))
        self.d_skip = nn.Parameter(torch.ones(self.n_heads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor, chunk_size: int) -> torch.Tensor:
        bsz, seq_len, _ = u.shape
        z, xbc, dt = self.in_proj(u).split(
            [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads], dim=-1
        )
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        x, bmat, cmat = xbc.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)  # (B,T,H)
        a = -self.a_log.exp() * dt
        x = x.view(bsz, seq_len, self.n_heads, self.head_dim)
        bmat = bmat[:, :, None, :].expand(bsz, seq_len, self.n_heads, self.d_state)
        cmat = cmat[:, :, None, :].expand(bsz, seq_len, self.n_heads, self.d_state)
        y = chunk_ssd(x * dt[..., None], a, bmat, cmat, chunk_size)
        y = y + x * self.d_skip[:, None]
        y = self.norm(y.reshape(bsz, seq_len, self.d_inner) * F.silu(z))
        return self.out_proj(y)
