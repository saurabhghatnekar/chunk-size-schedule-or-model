"""Kernel tier: the Triton kernels people actually train with, run on a CUDA GPU.

Importing this module is cheap; the libraries (which need Triton) are imported
only when a variant runs, so the file can be read and unit-tested without a GPU.

A case is one kernel. A variant is one (label, chunk length) pair; the first
variant that runs is the cross-chunk reference. Variants the kernel is expected
to reject (a chunk size outside what it supports) are listed on purpose: the
rejection is recorded as a row, because the range of chunk lengths a kernel
permits is itself a constraint on the study.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from ..archs.common import causal_softmax_attention, l2norm
from ..archs.delta import chunk_gated_delta
from ..archs.softmax import tiled_causal_softmax_attention
from ..library import vendor
from . import oracles

B, H, D = 2, 4, 64
SEQ_LENS = (1024, 1000)  # one length every chunk size divides, one that leaves a ragged tail


@dataclass(frozen=True)
class Variant:
    label: str
    chunk: int
    run: Callable  # (inputs on cuda, chunk) -> output tensor


@dataclass(frozen=True)
class KernelCase:
    name: str
    title: str
    implementation: str
    prior: str                      # "exact" | "boundary"
    mechanism: str
    make_inputs: Callable           # (generator, seq_len) -> dict of float64 cpu tensors
    variants: list
    oracle: Callable                # (float64 inputs, chunk) -> float64 output
    oracle_per_chunk: bool = False  # True when the reference definition itself depends on the chunk length
    role: str = "candidate"         # candidate | negative-control | sensitivity-control | baseline
    dtypes: tuple = ("float32", "bfloat16")
    seq_lens: tuple = SEQ_LENS
    cast: Callable | None = None    # (float64 inputs, dtype) -> inputs as the kernel expects them
    notes: str = ""


def _rand(gen, *shape):
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


# ---- inputs, kernel layout [B, T, H, D] ---------------------------------------------------

def delta_inputs(gen, t):
    return {"q": l2norm(_rand(gen, B, t, H, D)), "k": l2norm(_rand(gen, B, t, H, D)),
            "v": _rand(gen, B, t, H, D), "beta": torch.sigmoid(_rand(gen, B, t, H))}


def gated_inputs(gen, t):
    x = delta_inputs(gen, t)
    x["g"] = -0.05 * F.softplus(_rand(gen, B, t, H))
    return x


def gla_inputs(gen, t):
    x = delta_inputs(gen, t)
    del x["beta"]
    x["g"] = F.logsigmoid(_rand(gen, B, t, H, D) + 2.0) / 16
    return x


def simple_gla_inputs(gen, t):
    x = delta_inputs(gen, t)
    del x["beta"]
    x["g"] = -0.05 * F.softplus(_rand(gen, B, t, H))
    return x


def ttt_inputs(gen, t):
    return {"q": _rand(gen, B, t, H, D), "k": l2norm(_rand(gen, B, t, H, D)), "v": _rand(gen, B, t, H, D),
            "w": 1 + 0.1 * _rand(gen, H, D), "b": 0.1 * _rand(gen, H, D),
            "eta": 0.05 * torch.sigmoid(_rand(gen, B, t, H, 1))}


def lact_inputs(gen, t):
    n = B * H
    return {"w0": _rand(gen, n, D, D) / D ** 0.5, "w1": _rand(gen, n, D, D) / D ** 0.5,
            "w2": _rand(gen, n, D, D) / D ** 0.5,
            "q": l2norm(_rand(gen, n, t, D)), "k": l2norm(_rand(gen, n, t, D)), "v": _rand(gen, n, t, D),
            "lr0": 0.01 * F.softplus(_rand(gen, n, t, 1)), "lr1": 0.01 * F.softplus(_rand(gen, n, t, 1)),
            "lr2": 0.01 * F.softplus(_rand(gen, n, t, 1))}


def lact_cast(x, dtype):
    """As the LaCT docstring prescribes: q, k, v in the low-precision format, fast weights and learning rates in float32."""
    return {n: t.to(dtype if n in ("q", "k", "v") else torch.float32) for n, t in x.items()}


def softmax_inputs(gen, t):
    return {"q": _rand(gen, B, H, t, D), "k": _rand(gen, B, H, t, D), "v": _rand(gen, B, H, t, D)}


# ---- kernel calls (libraries imported lazily) ---------------------------------------------

def _fla(module, name):
    import importlib
    return getattr(importlib.import_module(f"fla.ops.{module}"), name)


def _delta(x, c):
    return _fla("delta_rule", "chunk_delta_rule")(x["q"], x["k"], x["v"], x["beta"], chunk_size=c)[0]


def _delta_rec(x, c):
    return _fla("delta_rule", "fused_recurrent_delta_rule")(x["q"], x["k"], x["v"], x["beta"])[0]


def _gated(x, c):
    return _fla("gated_delta_rule", "chunk_gated_delta_rule")(x["q"], x["k"], x["v"], x["g"], x["beta"], chunk_size=c)[0]


def _gated_rec(x, c):
    return _fla("gated_delta_rule", "fused_recurrent_gated_delta_rule")(x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"])[0]


def _simple(x, c):
    return _fla("simple_gla", "chunk_simple_gla")(x["q"], x["k"], x["v"], g=x["g"], chunk_size=c)[0]


def _simple_rec(x, c):
    return _fla("simple_gla", "fused_recurrent_simple_gla")(x["q"], x["k"], x["v"], g=x["g"])[0]


def _gla(x, c):
    return _fla("gla", "chunk_gla")(x["q"], x["k"], x["v"], x["g"])[0]


def _gla_rec(x, c):
    return _fla("gla", "fused_recurrent_gla")(x["q"], x["k"], x["v"], gk=x["g"])[0]


def _ttt(name):
    return lambda x, c: _fla("ttt", name)(x["q"], x["k"], x["v"], x["w"], x["b"], x["eta"], chunk_size=c)[0]


def _lact_torch(x, c):
    # decorators kept: the op runs under its own bfloat16 autocast (torch.compile is disabled by the environment)
    module = vendor.load("lact", "lact_llm/lact_model/ttt_operation.py", strip=False)
    return module.block_causal_lact_swiglu(x["w0"], x["w1"], x["w2"], x["q"], x["k"], x["v"],
                                           x["lr0"], x["lr1"], x["lr2"], chunk_size=c)


def _lact_fused(x, c):
    path = str(vendor.ensure("lact") / "lact_llm" / "lact_model")
    if path not in sys.path:
        sys.path.insert(0, path)
    import ttt_operation_fused_kernel as fused  # falls back to absolute imports of its Triton kernels
    return fused.postnorm_block_causal_lact_swiglu_fused_kernel_triton(
        x["w0"], x["w1"], x["w2"], x["q"], x["k"], x["v"], x["lr0"], x["lr1"], x["lr2"], chunk_size=c)


def _softmax_oracle(x, chunk):
    return causal_softmax_attention(x["q"], x["k"], x["v"])


def _ref_delta(perturb=None):
    def run(x, c):
        t = lambda a: a.transpose(1, 2)
        o = chunk_gated_delta(t(x["q"]) * D ** -0.5, t(x["k"]), t(x["v"]), t(x["beta"]), t(x["g"]), c, perturb=perturb)
        return t(o)
    return run


FLA = vendor.short("fla")
LACT = vendor.short("lact")

CASES = [
    # ---- controls, run on the same GPU in the same formats ----
    KernelCase(
        "gpu_softmax_tile", "Softmax attention: PyTorch SDPA and tiled streaming softmax", "torch + this repository",
        "exact", "tiling", softmax_inputs,
        [Variant("torch scaled_dot_product_attention", 0,
                 lambda x, c: F.scaled_dot_product_attention(x["q"], x["k"], x["v"], is_causal=True))]
        + [Variant("tiled", c, lambda x, c: tiled_causal_softmax_attention(x["q"], x["k"], x["v"], c)) for c in (16, 64, 256)],
        _softmax_oracle, role="negative-control",
    ),
    KernelCase(
        "gpu_ref_gated_delta", "Reference chunkwise Gated DeltaNet (this repository) on the GPU",
        "this repository, src/equivalence/archs/delta.py", "exact", "reassociation", gated_inputs,
        [Variant("chunkwise", c, _ref_delta()) for c in (16, 32, 64, 256)], oracles.delta_oracle, role="baseline",
        notes="What an exact chunkwise algorithm does in these formats: the yardstick for the kernels below.",
    ),
    KernelCase(
        "gpu_perturbed_leak_1e-2", "Reference Gated DeltaNet with carried state scaled by (1 + 1e-2) per chunk",
        "this repository", "exact", "reassociation", gated_inputs,
        [Variant("chunkwise", c, _ref_delta({"kind": "boundary_leak", "delta": 1e-2})) for c in (16, 32, 64)],
        oracles.delta_oracle, role="sensitivity-control",
    ),
    KernelCase(
        "gpu_perturbed_leak_1e-4", "Reference Gated DeltaNet with carried state scaled by (1 + 1e-4) per chunk",
        "this repository", "exact", "reassociation", gated_inputs,
        [Variant("chunkwise", c, _ref_delta({"kind": "boundary_leak", "delta": 1e-4})) for c in (16, 32, 64)],
        oracles.delta_oracle, role="sensitivity-control",
    ),
    # ---- exact family ----
    KernelCase(
        "fla_chunk_delta_rule", "DeltaNet Triton kernel", f"{FLA} fla.ops.delta_rule.chunk_delta_rule",
        "exact", "reassociation", delta_inputs,
        [Variant("fused_recurrent", 1, _delta_rec)] + [Variant("chunk", c, _delta) for c in (16, 32, 64, 128)],
        oracles.delta_oracle, dtypes=("float32", "bfloat16", "float16"),
    ),
    KernelCase(
        "fla_chunk_gated_delta_rule", "Gated DeltaNet Triton kernel",
        f"{FLA} fla.ops.gated_delta_rule.chunk_gated_delta_rule", "exact", "reassociation", gated_inputs,
        [Variant("fused_recurrent", 1, _gated_rec)] + [Variant("chunk", c, _gated) for c in (16, 32, 64, 128)],
        oracles.delta_oracle,
    ),
    KernelCase(
        "fla_chunk_simple_gla", "Scalar-decay linear attention Triton kernel (the Mamba-2 / RetNet recurrence)",
        f"{FLA} fla.ops.simple_gla.chunk_simple_gla", "exact", "reassociation", simple_gla_inputs,
        [Variant("fused_recurrent", 1, _simple_rec)] + [Variant("chunk", c, _simple) for c in (16, 32, 64, 128, 256, 48)],
        oracles.simple_gla_oracle,
    ),
    KernelCase(
        "fla_chunk_gla", "Gated linear attention Triton kernel", f"{FLA} fla.ops.gla.chunk_gla",
        "exact", "reassociation", gla_inputs,
        [Variant("fused_recurrent", 1, _gla_rec), Variant("chunk (size fixed by the kernel)", 64, _gla)],
        oracles.gla_oracle,
        notes="chunk_gla takes no chunk_size argument; it uses min(64, max(16, next_power_of_2(T))).",
    ),
    # ---- test-time training ----
    KernelCase(
        "fla_chunk_ttt_linear", "TTT-Linear Triton kernel", f"{FLA} fla.ops.ttt.chunk_ttt_linear",
        "boundary", "inner-loop boundary", ttt_inputs,
        [Variant("chunk", c, _ttt("chunk_ttt_linear")) for c in (16, 64, 256)], oracles.ttt_oracle, oracle_per_chunk=True,
    ),
    KernelCase(
        "fla_fused_chunk_ttt_linear", "TTT-Linear fused Triton kernel", f"{FLA} fla.ops.ttt.fused_chunk_ttt_linear",
        "boundary", "inner-loop boundary", ttt_inputs,
        [Variant("fused_chunk", c, _ttt("fused_chunk_ttt_linear")) for c in (16, 64, 256)], oracles.ttt_oracle,
        oracle_per_chunk=True,
    ),
    KernelCase(
        "lact_block_causal_swiglu_autocast", "LaCT official op on CUDA under its own bfloat16 autocast",
        f"{LACT} lact_llm/lact_model/ttt_operation.py block_causal_lact_swiglu", "boundary", "inner-loop boundary",
        lact_inputs, [Variant("apply-then-update", c, _lact_torch) for c in (64, 256, 1024)], oracles.lact_oracle,
        oracle_per_chunk=True, dtypes=("bfloat16",), cast=lact_cast,
        notes="q, k, v in bfloat16; fast weights and learning rates in float32, as the op's docstring prescribes.",
    ),
    KernelCase(
        "lact_fused_triton_kernel", "LaCT fused Triton kernel (post-norm)",
        f"{LACT} lact_llm/lact_model/ttt_operation_fused_kernel.py postnorm_block_causal_lact_swiglu_fused_kernel_triton",
        "boundary", "inner-loop boundary", lact_inputs,
        [Variant("fused", c, _lact_fused) for c in (64, 256, 1024)], oracles.lact_oracle, oracle_per_chunk=True,
        dtypes=("bfloat16",), cast=lact_cast,
        notes="Oracle is the plain post-norm op at the same chunk length.",
    ),
]
