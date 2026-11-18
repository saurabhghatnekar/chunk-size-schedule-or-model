"""Shared building blocks for the reference mixers.

Every mixer in this package is dtype-generic (it runs unchanged in bfloat16,
float16, float32 and float64) and takes the chunk length as a *call-time*
argument: `mixer(x, chunk_size)`. The parameters never depend on the chunk
length, so one checkpoint serves every chunk length in a sweep.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.pow(2).sum(-1, keepdim=True) + eps).sqrt()


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, H*D) -> (B, H, T, D)"""
    b, t, hd = x.shape
    return x.view(b, t, n_heads, hd // n_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, D) -> (B, T, H*D)"""
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)


def chunk_bounds(seq_len: int, chunk_size: int):
    """Yield (start, end) for contiguous chunks; the last chunk may be ragged."""
    for s in range(0, seq_len, chunk_size):
        yield s, min(s + chunk_size, seq_len)


def causal_softmax_attention(q, k, v, window: int | None = None):
    """Plain causal softmax attention, optionally banded. q,k,v: (B,H,T,D)."""
    t = q.shape[-2]
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    idx = torch.arange(t, device=q.device)
    allowed = idx[None, :] <= idx[:, None]
    if window is not None:
        allowed = allowed & (idx[:, None] - idx[None, :] < window)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return F.softmax(scores, dim=-1) @ v
