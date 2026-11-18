"""The equivalence test: one checkpoint, one batch, forward and backward at several
chunk lengths and precisions.

    uv run python -m src.equivalence.test                 # everything -> results/equivalence.csv
    uv run python -m src.equivalence.test --arch gla lact # a subset (does not overwrite the csv)

No training is involved. Each row holds the deviation of one chunk length from the
reference chunk length (the smallest the architecture supports) for one precision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .archs import SPECS
from .core import DTYPES, run_arch
from .model import ROOT, ModelConfig

RESULTS = ROOT / "results"
COLUMNS = [
    "tier", "architecture", "role", "implementation", "prior", "mechanism", "dtype", "unit_roundoff",
    "output_dtype", "variant", "chunk", "ref_variant", "ref_chunk", "status", "fwd_max_rel", "fwd_l2_rel", "fwd_max_abs", "grad_max_rel", "grad_l2_rel",
    "grad_max_rel_per_tensor", "grad_worst_tensor", "loss", "seconds", "seq_len", "batch", "d_model",
    "n_layers", "checkpoint_sha256", "batch_sha256", "torch", "platform", "run_utc",
]


def environment() -> dict:
    return {
        "torch": torch.__version__,
        "platform": f"{platform.system()}-{platform.machine()}-cpu",
        "run_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", nargs="*", default=None, help="subset of architectures (default: all)")
    parser.add_argument("--dtype", nargs="*", default=list(DTYPES), choices=list(DTYPES))
    parser.add_argument("--tier", choices=["all", "reference", "library"], default="all",
                        help="reference: the implementations in this repository; library: pinned third-party code")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = ModelConfig()
    names = args.arch or list(SPECS)
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        parser.error(f"unknown architecture(s) {unknown}; choose from {list(SPECS)}")
    env = environment()
    print(f"equivalence test  {asdict(cfg)}  torch {env['torch']}  {env['platform']}")
    rows = []
    if args.tier in ("all", "reference"):
        for name in names:
            rows += [{**r, **env} for r in run_arch(SPECS[name], cfg, args.dtype)]
    if args.tier in ("all", "library") and args.arch is None:
        from .library.run import run_library  # imported late: it fetches the pinned third-party sources
        print("library tier (pinned third-party reference ops, pure PyTorch, CPU)")
        rows += [{**r, **env} for r in run_library()]
    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)

    full = args.arch is None and args.tier == "all" and set(args.dtype) == set(DTYPES)
    out = args.out or (RESULTS / "equivalence.csv" if full else None)
    if out is None:
        print("partial run: nothing written (pass --out to save)")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}  ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
