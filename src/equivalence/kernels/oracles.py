"""Float64 oracles for the kernel tier, in the kernels' own tensor layout [B, T, H, D].

A Triton kernel has no float64 path, so the precision ladder used on CPU is not
available. Instead every kernel output is compared with the same function
evaluated in float64 on the same (already rounded) inputs. For the exact family
the oracle is the token-by-token recurrence and does not depend on the chunk
length; for the test-time-training family the oracle is the reference
definition at the same chunk length.

The conventions (scaling, gate placement) are checked against the libraries' own
reference ops in tests/theory/test_oracle_conventions.py.
"""
from __future__ import annotations

import torch

from ..archs.delta import recurrent_gated_delta
from ..archs.gla import recurrent_gla
from ..archs.lact import block_causal_lact_swiglu
from ..library import vendor


def _t(x):
    return x.transpose(1, 2)


def delta_oracle(x: dict, chunk: int) -> torch.Tensor:
    q, k, v, beta = _t(x["q"]), _t(x["k"]), _t(x["v"]), _t(x["beta"])
    g = _t(x["g"]) if "g" in x else torch.zeros_like(beta)
    return _t(recurrent_gated_delta(q * k.shape[-1] ** -0.5, k, v, beta, g))


def gla_oracle(x: dict, chunk: int) -> torch.Tensor:
    q, k, v, g = _t(x["q"]), _t(x["k"]), _t(x["v"]), _t(x["g"])
    return _t(recurrent_gla(q * k.shape[-1] ** -0.5, k, v, g))


def simple_gla_oracle(x: dict, chunk: int) -> torch.Tensor:
    q, k, v = _t(x["q"]), _t(x["k"]), _t(x["v"])
    g = _t(x["g"])[..., None].expand_as(k)
    return _t(recurrent_gla(q * k.shape[-1] ** -0.5, k, v, g))


def ttt_oracle(x: dict, chunk: int) -> torch.Tensor:
    module = vendor.load("fla", "fla/ops/ttt/naive.py", lift=True)
    with vendor.default_dtype(torch.float64):
        # the reference op scales q in place
        return module.chunk_ttt_linear_ref(
            x["q"].clone(), x["k"], x["v"], x["w"], x["b"], x["eta"], mini_batch_size=chunk)[0]


def lact_oracle(x: dict, chunk: int) -> torch.Tensor:
    return block_causal_lact_swiglu(x["w0"], x["w1"], x["w2"], x["q"], x["k"], x["v"],
                                    x["lr0"], x["lr1"], x["lr2"], chunk)
