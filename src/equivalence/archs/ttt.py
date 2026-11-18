"""Test-time-training memory with mini-batch gradient descent (Titans, arXiv 2501.00663;
TTT layers, arXiv 2407.04620), linear memory.

Per head, memory M in R^{K x V}, inner loss l(M; k, v) = ||k M - v||^2, momentum
buffer S, and per-token gates theta_t (inner learning rate), eta_t (momentum)
and alpha_t (forgetting):

    S_t = eta_t S_{t-1} - theta_t * grad l(M_c ; k_t, v_t)
    M_t = (1 - alpha_t) M_{t-1} + S_t
    o_t = q_t M_t

The chunk length enters through M_c: every gradient inside a chunk is evaluated
at the memory as it stood when the chunk began. With chunk length 1, M_c = M_{t-1}
and this is online gradient descent (for eta = alpha = 0 it is exactly the delta
rule with beta = 2 theta). With one chunk spanning the sequence, M_c = M_0 for
every token. Those are different functions of the input, so the chunk length is
part of the model here and not a way of scheduling one computation.

The implementation is a plain token loop: it is the definition, written to be
obviously right rather than fast.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, l2norm, merge_heads, split_heads


def minibatch_ttt_linear(q, k, v, theta, eta, alpha, chunk_size: int):
    """q,k: (B,H,T,K); v: (B,H,T,V); theta,eta,alpha: (B,H,T)."""
    b, h, t, kd = q.shape
    memory = q.new_zeros(b, h, kd, v.shape[-1])
    momentum = torch.zeros_like(memory)
    frozen = memory
    outs = []
    for i in range(t):
        if i % chunk_size == 0:
            frozen = memory  # chunk boundary: refresh the weights the gradients see
        ki = k[:, :, i, None, :]  # (B,H,1,K)
        grad = 2.0 * ki.transpose(-1, -2) @ (ki @ frozen - v[:, :, i, None, :])
        momentum = eta[:, :, i, None, None] * momentum - theta[:, :, i, None, None] * grad
        memory = (1.0 - alpha[:, :, i, None, None]) * memory + momentum
        outs.append((q[:, :, i, None, :] @ memory).squeeze(-2))
    return torch.stack(outs, dim=2)


class TTTMixer(nn.Module):
    name = "ttt"

    def __init__(self, d_model: int, n_heads: int, max_lr: float = 0.5):
        super().__init__()
        self.n_heads = n_heads
        self.max_lr = max_lr
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 3 * n_heads)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model // n_heads)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        q = l2norm(split_heads(self.q_proj(x), self.n_heads))
        k = l2norm(split_heads(self.k_proj(x), self.n_heads))
        v = split_heads(self.v_proj(x), self.n_heads)
        theta, eta, alpha = self.gate_proj(x).transpose(1, 2).chunk(3, dim=1)  # each (B,H,T)
        theta = self.max_lr * theta.sigmoid()
        eta = (eta - 1.0).sigmoid()
        alpha = (alpha - 4.0).sigmoid()
        o = minibatch_ttt_linear(q, k, v, theta, eta, alpha, chunk_size)
        o = merge_heads(self.norm(o))
        return self.o_proj(o * F.silu(self.g_proj(x)))
