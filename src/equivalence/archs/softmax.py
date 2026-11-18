"""Negative control: causal softmax attention evaluated in key tiles of varying size.

The tile size is a pure kernel-scheduling choice. Keys are consumed tile by tile
with the streaming ("online") softmax used by blocked attention kernels: a
running maximum m, a running normaliser l and an unnormalised accumulator are
rescaled whenever a tile raises the maximum. Different tile sizes re-associate
the normalising sum but compute the same function, so the outputs and gradients
must agree to machine precision. If they do not, the harness is wrong.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import merge_heads, split_heads


def tiled_causal_softmax_attention(q, k, v, tile: int):
    """q,k,v: (B,H,T,D). Streaming softmax over key tiles of size `tile`."""
    bsz, h, t, d = q.shape
    q = q * (d ** -0.5)
    pos = torch.arange(t, device=q.device)
    run_max = q.new_full((bsz, h, t, 1), float("-inf"))
    run_sum = q.new_zeros(bsz, h, t, 1)
    acc = q.new_zeros(bsz, h, t, v.shape[-1])
    for s in range(0, t, tile):
        e = min(s + tile, t)
        scores = q @ k[:, :, s:e].transpose(-1, -2)  # (B,H,T,tile)
        allowed = pos[s:e][None, :] <= pos[:, None]  # (T,tile)
        scores = scores.masked_fill(~allowed, float("-inf"))
        new_max = torch.maximum(run_max, scores.amax(dim=-1, keepdim=True))
        # rows that have not seen an allowed key yet keep max = -inf; guard the rescale
        safe_max = torch.where(torch.isinf(new_max), torch.zeros_like(new_max), new_max)
        scale_old = torch.where(torch.isinf(run_max), torch.zeros_like(run_max), (run_max - safe_max).exp())
        p = (scores - safe_max).exp()  # exp(-inf) = 0 for masked keys
        run_sum = run_sum * scale_old + p.sum(dim=-1, keepdim=True)
        acc = acc * scale_old + p @ v[:, :, s:e]
        run_max = new_max
    return acc / run_sum


class SoftmaxTileMixer(nn.Module):
    name = "softmax"

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        q = split_heads(self.q_proj(x), self.n_heads)
        k = split_heads(self.k_proj(x), self.n_heads)
        v = split_heads(self.v_proj(x), self.n_heads)
        return self.o_proj(merge_heads(tiled_causal_softmax_attention(q, k, v, chunk_size)))
