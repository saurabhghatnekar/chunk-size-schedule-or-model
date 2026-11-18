"""Library tier: the same test applied to code other people ship.

Only pure-PyTorch code paths are covered here, because this machine has no CUDA
and therefore no Triton. Each case is op-level: fixed random inputs, the output,
and the gradient of a fixed linear probe of the output with respect to every
floating-point input. A variant is one (implementation, chunk length) pair and
the first variant of a case is its reference.

The Triton kernels (`chunk_gla`, `chunk_delta_rule`, `chunk_gated_delta_rule`,
the fused LaCT kernel) are what people actually train with and must be put
through the same procedure on a GPU. See docs/classification.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..archs.common import l2norm
from . import vendor

T, B, H, D = 256, 2, 2, 32


@dataclass(frozen=True)
class Variant:
    label: str      # e.g. "recurrent", "chunkwise"
    chunk: int
    run: Callable   # (module, inputs) -> output tensor


@dataclass(frozen=True)
class LibCase:
    name: str
    title: str
    repo: str
    relpath: str
    prior: str
    mechanism: str
    make_inputs: Callable   # generator -> dict of float64 tensors
    variants: list
    inject: Callable | None = None   # lift -> dict of names for intra-package imports
    role: str = "candidate"        # "implementation-consistency": two implementations at one chunk length

    @property
    def implementation(self) -> str:
        return f"{vendor.short(self.repo)} {self.relpath}"


def _rand(gen, *shape):
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


# ---- inputs ------------------------------------------------------------------------------

def delta_inputs(gen):  # layout (B,H,T,D)
    return {"q": l2norm(_rand(gen, B, H, T, D)), "k": l2norm(_rand(gen, B, H, T, D)),
            "v": _rand(gen, B, H, T, D), "beta": torch.sigmoid(_rand(gen, B, H, T))}


def gated_inputs(gen):  # layout (B,T,H,D)
    return {"q": l2norm(_rand(gen, B, T, H, D)), "k": l2norm(_rand(gen, B, T, H, D)),
            "v": _rand(gen, B, T, H, D), "beta": torch.sigmoid(_rand(gen, B, T, H)),
            "g": -0.1 * torch.nn.functional.softplus(_rand(gen, B, T, H))}


def ttt_inputs(gen):
    return {"q": _rand(gen, B, T, H, D), "k": l2norm(_rand(gen, B, T, H, D)), "v": _rand(gen, B, T, H, D),
            "w": 1 + 0.1 * _rand(gen, H, D), "b": 0.1 * _rand(gen, H, D),
            "eta": 0.1 * torch.sigmoid(_rand(gen, B, T, H, 1))}


def titans_inputs(gen):
    base = ttt_inputs(gen)
    base["theta"] = 0.02 * torch.sigmoid(_rand(gen, B, T, H, 1))
    base["alpha"] = 0.05 * torch.sigmoid(_rand(gen, B, T, H, 1))
    base["eta"] = torch.sigmoid(_rand(gen, B, T, H, 1) - 1)
    return base


def lact_inputs(gen):
    n = B * H
    return {"w0": _rand(gen, n, D, D) / D ** 0.5, "w1": _rand(gen, n, D, D) / D ** 0.5,
            "w2": _rand(gen, n, D, D) / D ** 0.5,
            "q": l2norm(_rand(gen, n, T, D)), "k": l2norm(_rand(gen, n, T, D)), "v": _rand(gen, n, T, D),
            "lr0": 0.01 * torch.nn.functional.softplus(_rand(gen, n, T, 1)),
            "lr1": 0.01 * torch.nn.functional.softplus(_rand(gen, n, T, 1)),
            "lr2": 0.01 * torch.nn.functional.softplus(_rand(gen, n, T, 1))}


# ---- cases -------------------------------------------------------------------------------

def _titans_inject(lift: bool):
    log_impl = vendor.load("fla", "fla/ops/titans/log_impl.py", lift=lift)
    return {"combine_params_log": log_impl.combine_params_log}


def _titans(use_chunk):
    return lambda m, x, c: m.chunk_titans_linear_ref(
        x["q"], x["k"], x["v"], x["w"], x["b"], x["theta"], x["alpha"], x["eta"],
        chunk_size=c, use_chunk=use_chunk)[0]


CASES = [
    LibCase(
        "fla_delta_rule", "DeltaNet reference ops", "fla", "fla/ops/delta_rule/naive.py",
        "exact", "reassociation", delta_inputs,
        [Variant("recurrent", 1, lambda m, x, c: m.delta_rule_recurrence(x["q"], x["k"], x["v"], x["beta"])[0])]
        + [Variant("chunkwise", c, lambda m, x, c: m.delta_rule_chunkwise(x["q"], x["k"], x["v"], x["beta"], chunk_size=c)[0])
           for c in (1, 4, 16, 64, 256)],
    ),
    LibCase(
        "fla_gated_delta_rule", "Gated DeltaNet reference ops", "fla", "fla/ops/gated_delta_rule/naive.py",
        "exact", "reassociation", gated_inputs,
        [Variant("recurrent", 1, lambda m, x, c: m.naive_recurrent_gated_delta_rule(x["q"], x["k"], x["v"], x["beta"], x["g"])[0])]
        + [Variant("chunkwise", c, lambda m, x, c: m.naive_chunk_gated_delta_rule(x["q"], x["k"], x["v"], x["g"], x["beta"], chunk_size=c)[0])
           for c in (1, 4, 16, 64, 256, 48, 100)],
    ),
    LibCase(
        "fla_simple_gla", "Scalar-decay linear attention reference ops (the Mamba-2 / RetNet recurrence)",
        "fla", "fla/ops/simple_gla/naive.py", "exact", "reassociation", gated_inputs,
        [Variant("recurrent", 1, lambda m, x, c: m.naive_recurrent_simple_gla(x["q"], x["k"], x["v"], x["g"])[0])]
        + [Variant("chunkwise", c, lambda m, x, c: m.naive_chunk_simple_gla(x["q"], x["k"], x["v"], x["g"], chunk_size=c)[0])
           for c in (1, 4, 16, 64, 256, 48, 100)],
    ),
    LibCase(
        "fla_ttt_linear", "TTT-Linear reference op (mini-batch size)", "fla", "fla/ops/ttt/naive.py",
        "boundary", "inner-loop boundary", ttt_inputs,
        [Variant("mini-batch", c, lambda m, x, c: m.chunk_ttt_linear_ref(
            x["q"], x["k"], x["v"], x["w"], x["b"], x["eta"], mini_batch_size=c)[0]) for c in (1, 4, 16, 64, 256, 48)],
    ),
    LibCase(
        "fla_titans_sequential", "Titans reference op, token loop", "fla", "fla/ops/titans/naive.py",
        "boundary", "inner-loop boundary", titans_inputs,
        [Variant("sequential", c, _titans(False)) for c in (1, 4, 16, 64, 256, 48)],
        inject=_titans_inject,
    ),
    LibCase(
        "fla_titans_chunkwise", "Titans reference op, closed-form per chunk", "fla", "fla/ops/titans/naive.py",
        "boundary", "inner-loop boundary", titans_inputs,
        [Variant("chunkwise", c, _titans(True)) for c in (4, 16, 64, 256, 48)],
        inject=_titans_inject,
    ),
    LibCase(
        "fla_titans_loop_vs_closed_form", "Titans reference op: token loop against closed form at one chunk length",
        "fla", "fla/ops/titans/naive.py", "exact", "reassociation", titans_inputs,
        [Variant("sequential", 16, _titans(False)), Variant("chunkwise", 16, _titans(True))],
        inject=_titans_inject, role="implementation-consistency",
    ),
    LibCase(
        "lact_block_causal_swiglu", "LaCT official block-causal SwiGLU op", "lact",
        "lact_llm/lact_model/ttt_operation.py", "boundary", "inner-loop boundary", lact_inputs,
        [Variant("apply-then-update", c, lambda m, x, c: m.block_causal_lact_swiglu(
            x["w0"], x["w1"], x["w2"], x["q"], x["k"], x["v"], x["lr0"], x["lr1"], x["lr2"], chunk_size=c))
         for c in (1, 4, 16, 64, 256, 48)],
    ),
]
