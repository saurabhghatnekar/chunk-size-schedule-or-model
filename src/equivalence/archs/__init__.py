"""Registry of the reference architectures the equivalence test covers.

`prior` is the expectation read off the mathematics *before* the test runs:
"exact" means the chunkwise form is a re-association of one recurrence and must
agree across chunk lengths up to rounding; "boundary" means a named mechanism
makes the chunk boundary part of the computed function. The test does not use
`prior` to decide whether chunk length changes the function. It uses it only
afterwards, to tell an expected structural difference (architectural) from an
unexpected one (implementation bug).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

from .delta import DeltaMixer
from .flash import FlashMixer
from .gla import GLAMixer
from .hla import HLAMixer
from .lact import LaCTMixer
from .mamba2 import Mamba2Mixer
from .softmax import SoftmaxTileMixer
from .ttt import TTTMixer

FULL = [1, 4, 16, 64, 256, 512, 48, 100]  # powers of four up to the sequence, plus two ragged tilings


@dataclass(frozen=True)
class ArchSpec:
    name: str
    title: str
    paper: str
    build: Callable[[int, int], nn.Module]
    prior: str  # "exact" | "boundary"
    mechanism: str  # reassociation | inner-loop boundary | attention-pattern boundary | routing boundary | tiling
    role: str = "candidate"  # candidate | negative-control | sensitivity-control
    chunk_lengths: list[int] = field(default_factory=lambda: list(FULL))


SPECS: dict[str, ArchSpec] = {
    s.name: s
    for s in [
        ArchSpec(
            "softmax_tile", "Softmax transformer, tiled streaming softmax", "control",
            lambda d, h: SoftmaxTileMixer(d, h), "exact", "tiling", role="negative-control",
        ),
        ArchSpec(
            "gla", "Gated linear attention", "arXiv 2312.06635",
            lambda d, h: GLAMixer(d, h), "exact", "reassociation",
        ),
        ArchSpec(
            "deltanet", "DeltaNet", "arXiv 2406.06484",
            lambda d, h: DeltaMixer(d, h, gated=False), "exact", "reassociation",
        ),
        ArchSpec(
            "gated_deltanet", "Gated DeltaNet", "arXiv 2412.06464",
            lambda d, h: DeltaMixer(d, h, gated=True), "exact", "reassociation",
        ),
        ArchSpec(
            "mamba2", "Mamba-2 SSD", "arXiv 2405.21060",
            lambda d, h: Mamba2Mixer(d, h), "exact", "reassociation",
        ),
        ArchSpec(
            "ttt_titans", "Test-time training, Titans-style mini-batch memory", "arXiv 2501.00663",
            lambda d, h: TTTMixer(d, h), "boundary", "inner-loop boundary",
        ),
        ArchSpec(
            "lact", "Test-time training, LaCT-style block-causal SwiGLU", "arXiv 2505.23884",
            lambda d, h: LaCTMixer(d, h), "boundary", "inner-loop boundary",
        ),
        ArchSpec(
            "flash", "FLASH mixed chunk attention", "arXiv 2202.10447",
            lambda d, h: FlashMixer(d, h), "boundary", "attention-pattern boundary",
        ),
        ArchSpec(
            "hla", "HLA-style chunk routing over Gated DeltaNet", "arXiv 2610.05842",
            lambda d, h: HLAMixer(d, h), "boundary", "routing boundary",
            chunk_lengths=[16, 32, 64, 128, 256, 512, 48, 100],
        ),
        # Sensitivity controls: a provably exact kernel with a deliberate defect.
        ArchSpec(
            "perturbed_leak_1e-3", "Gated DeltaNet with carried state scaled by (1 + 1e-3) per chunk", "control",
            lambda d, h: DeltaMixer(d, h, gated=True, perturb={"kind": "boundary_leak", "delta": 1e-3}),
            "exact", "reassociation", role="sensitivity-control",
        ),
        ArchSpec(
            "perturbed_leak_1e-6", "Gated DeltaNet with carried state scaled by (1 + 1e-6) per chunk", "control",
            lambda d, h: DeltaMixer(d, h, gated=True, perturb={"kind": "boundary_leak", "delta": 1e-6}),
            "exact", "reassociation", role="sensitivity-control",
        ),
        ArchSpec(
            "perturbed_leak_1e-9", "Gated DeltaNet with carried state scaled by (1 + 1e-9) per chunk", "control",
            lambda d, h: DeltaMixer(d, h, gated=True, perturb={"kind": "boundary_leak", "delta": 1e-9}),
            "exact", "reassociation", role="sensitivity-control",
        ),
        ArchSpec(
            "perturbed_stale_gate", "Gated DeltaNet whose carried state misses the last token's gate", "control",
            lambda d, h: DeltaMixer(d, h, gated=True, perturb={"kind": "stale_gate"}),
            "exact", "reassociation", role="sensitivity-control",
        ),
    ]
}
