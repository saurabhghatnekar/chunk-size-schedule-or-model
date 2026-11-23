"""FLASH mixed chunk attention, causal variant (Hua et al., ICML 2022, arXiv 2202.10447).

Follows the paper's pseudo-code for the FLASH block (Code 8 in its appendix), checked against the listing.
With chunk (group) size g, a shared low-dimensional z = silu(W x) is turned into four per-dimension scaled
and shifted copies (quad_q, quad_k, lin_q, lin_k), and

    quadratic_t = sum_{m in chunk(t), m <= t} relu( quad_q_t . quad_k_m / g + bias[t - m] )^2  v_m
    linear_t    = lin_q_t . mean_{chunks c' < chunk(t)} ( sum_{m in c'} lin_k_m^T v_m / g )
    out         = W_o ( u * (quadratic + linear) )

The chunk boundary decides which token pairs interact through the quadratic kernel and which only through
the linear summary, and g also appears as a normaliser in both paths, so the chunk length is architectural
by construction.

Differences from the listing, none of which involves the chunk boundary: no rotary embedding is applied to
the shared representation; the scale parameters are initialised around one; the relative-position bias is
one learned scalar per offset from a fixed-size table, so the parameter count does not depend on the chunk
length; and u, v and z come from three linear layers where the listing splits one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashMixer(nn.Module):
    name = "flash"

    def __init__(self, d_model: int, n_heads: int, s_dim: int = 32, expansion: int = 2, max_offset: int = 4096):
        super().__init__()
        del n_heads  # single-head by design (gated attention unit)
        e_dim = expansion * d_model
        self.to_u = nn.Linear(d_model, e_dim)
        self.to_v = nn.Linear(d_model, e_dim)
        self.to_z = nn.Linear(d_model, s_dim)
        self.gamma = nn.Parameter(torch.ones(4, s_dim) + 0.02 * torch.randn(4, s_dim))
        self.beta = nn.Parameter(0.02 * torch.randn(4, s_dim))
        self.rel_bias = nn.Parameter(0.02 * torch.randn(max_offset))
        self.o_proj = nn.Linear(e_dim, d_model)

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        g = chunk_size
        u = F.silu(self.to_u(x))
        v = F.silu(self.to_v(x))
        z = F.silu(self.to_z(x))
        pad = (-seq_len) % g
        if pad:
            v = F.pad(v, (0, 0, 0, pad))
            z = F.pad(z, (0, 0, 0, pad))
        n_groups = (seq_len + pad) // g
        v = v.view(bsz, n_groups, g, -1)
        z = z.view(bsz, n_groups, g, -1)
        quad_q, quad_k, lin_q, lin_k = (z[..., None, :] * self.gamma + self.beta).unbind(dim=-2)
        if pad:
            # padded keys must not contribute to the cross-chunk summary
            valid = (torch.arange(seq_len + pad, device=x.device) < seq_len).view(1, n_groups, g, 1)
            lin_k = lin_k * valid

        # quadratic attention inside each chunk
        idx = torch.arange(g, device=x.device)
        offset = idx[:, None] - idx[None, :]
        causal = offset >= 0
        bias = self.rel_bias[offset.clamp(min=0)]
        kernel = F.relu(quad_q @ quad_k.transpose(-1, -2) / g + bias).pow(2)
        quadratic = kernel.masked_fill(~causal, 0.0) @ v

        # linear attention across chunks: the mean of the summaries of all strictly earlier chunks
        # (Code 8: the causal segment mask is row-normalised, so earlier chunks are averaged, not summed)
        lin_kv = lin_k.transpose(-1, -2) @ v / g  # (B,G,S,E)
        earlier = torch.arange(n_groups, device=x.device, dtype=lin_kv.dtype).clamp(min=1).view(1, -1, 1, 1)
        lin_kv = (lin_kv.cumsum(dim=1) - lin_kv) / earlier
        linear = lin_q @ lin_kv

        out = (quadratic + linear).reshape(bsz, seq_len + pad, -1)[:, :seq_len]
        return self.o_proj(u * out)
