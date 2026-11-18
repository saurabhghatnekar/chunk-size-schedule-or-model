"""LaCT-style large-chunk test-time training (Zhang et al., arXiv 2505.23884).

A dtype-generic re-statement of `block_causal_lact_swiglu` from the official
repository (lact_llm/lact_model/ttt_operation.py), without Muon and without
momentum. The fast weights are a SwiGLU MLP

    f(x) = w1 (silu(w0 x) * (w2 x)),

the inner loss is the negative dot product -<f(k), v> weighted by a per-token
learning rate, and the layer is "apply then update": every query in a chunk is
answered with the fast weights as they stood when the chunk began, then one
gradient step is taken on the whole chunk and each fast-weight row is rescaled
to its original norm.

Two details are copied from the reference on purpose because they are chunk
dependent: the loop runs over `range(0, T - chunk, chunk)`, so the final chunk
(a full one included) is applied but never used for an update, and a chunk
length of at least the sequence length means the fast weights are never updated
at all. As in the reference layer, a causal sliding-window softmax attention
with a window independent of the chunk length is added to the output.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, causal_softmax_attention, l2norm, split_heads


def _silu_backprop(dy, x):
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


def _apply(w0, w1, w2, q):
    """q: (b, dk, l) -> (b, dv, l)"""
    return w1 @ (F.silu(w0 @ q) * (w2 @ q))


def block_causal_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk_size: int):
    """w0,w2: (b,dh,dk); w1: (b,dv,dh); q,k: (b,l,dk); v: (b,l,dv); lr*: (b,l,1)."""
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)
    seq_len = k.shape[1]
    outs = []
    e_index = 0
    for s_index in range(0, seq_len - chunk_size, chunk_size):
        e_index = s_index + chunk_size
        ki = k[:, s_index:e_index, :]
        vi = v[:, :, s_index:e_index]
        outs.append(_apply(w0, w1, w2, q[:, :, s_index:e_index]))

        gate_before_act = w0 @ ki.transpose(1, 2)
        hidden_before_mul = w2 @ ki.transpose(1, 2)
        hidden = F.silu(gate_before_act) * hidden_before_mul
        dhidden = w1.transpose(1, 2) @ vi
        dhidden_before_mul = dhidden * F.silu(gate_before_act)
        dgate_before_act = _silu_backprop(dhidden * hidden_before_mul, gate_before_act)

        dw1 = vi @ (hidden.transpose(1, 2) * lr1[:, s_index:e_index, :])
        dw0 = dgate_before_act @ (ki * lr0[:, s_index:e_index, :])
        dw2 = dhidden_before_mul @ (ki * lr2[:, s_index:e_index, :])

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # final chunk: applied with the last fast weights, never used for an update
    outs.append(_apply(w0, w1, w2, q[:, :, e_index:seq_len]))
    return torch.cat(outs, dim=2).transpose(1, 2)


class LaCTMixer(nn.Module):
    name = "lact"

    def __init__(self, d_model: int, n_heads: int, window: int = 32, base_lr: float = 0.01):
        super().__init__()
        self.n_heads = n_heads
        self.window = window
        head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w0 = nn.Parameter(torch.randn(n_heads, head_dim, head_dim) / math.sqrt(head_dim))
        self.w1 = nn.Parameter(torch.randn(n_heads, head_dim, head_dim) / math.sqrt(head_dim))
        self.w2 = nn.Parameter(torch.randn(n_heads, head_dim, head_dim) / math.sqrt(head_dim))
        self.lr_proj = nn.Linear(d_model, 3 * n_heads)
        self.base_lr_inv = math.log(math.expm1(base_lr))
        self.qk_scale = nn.Parameter(torch.ones(d_model, 2))
        self.qk_offset = nn.Parameter(torch.zeros(d_model, 2))
        self.ttt_norm = RMSNorm(head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        h = self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = causal_softmax_attention(
            split_heads(q, h), split_heads(k, h), split_heads(v, h), window=self.window
        ).transpose(1, 2).reshape(bsz, seq_len, d_model)

        fast_q = q * self.qk_scale[:, 0] + self.qk_offset[:, 0]
        fast_k = k * self.qk_scale[:, 1] + self.qk_offset[:, 1]

        def to_fw(t):  # (B,T,H*D) -> (B*H,T,D)
            return split_heads(t, h).reshape(bsz * h, seq_len, -1)

        fast_q = l2norm(F.silu(to_fw(fast_q)), eps=1e-10)
        fast_k = l2norm(F.silu(to_fw(fast_k)), eps=1e-10)
        fast_v = F.silu(to_fw(v))
        lr = F.softplus(self.lr_proj(x) + self.base_lr_inv)  # (B,T,3H)
        lr = lr.view(bsz, seq_len, h, 3).transpose(1, 2).reshape(bsz * h, seq_len, 3)
        lr0, lr1, lr2 = lr.split(1, dim=-1)
        fw = block_causal_lact_swiglu(
            self.w0.repeat(bsz, 1, 1),
            self.w1.repeat(bsz, 1, 1),
            self.w2.repeat(bsz, 1, 1),
            fast_q, fast_k, fast_v, lr0, lr1, lr2, chunk_size,
        )
        fw = self.ttt_norm(fw).view(bsz, h, seq_len, -1).transpose(1, 2).reshape(bsz, seq_len, d_model)
        return self.o_proj(attn + fw)
