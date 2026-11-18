"""Controls that must pass before any classification is read.

1. Negative control. A softmax transformer evaluated at varying kernel tile size
   must agree to machine precision. This validates the harness: if a pure
   scheduling parameter shows a deviation, the harness is broken.
2. Sensitivity controls. A provably exact kernel (Gated DeltaNet) with a
   deliberate defect must be flagged. This validates that the test can see a
   difference when there is one, and shows how small a defect it can see.

    uv run python -m src.equivalence.controls   # reads results/equivalence.csv, exits 1 on failure
"""
from __future__ import annotations

import sys

import pandas as pd

from .archs import SPECS
from .core import EPS
from .model import ROOT
from .precision_scaling import summarise

# "machine precision" for the negative control: this many unit roundoffs, per format
NEGATIVE_CONTROL_ULPS = 1e3


def check(frame: pd.DataFrame) -> tuple[bool, list[str]]:
    lines, passed = [], True
    summary = summarise(frame).set_index("architecture")

    neg = [n for n, s in SPECS.items() if s.role == "negative-control"]
    for name in neg:
        for dtype in ["float32", "float64"]:
            dev = max(summary.loc[name, f"fwd_dev_{dtype}"], summary.loc[name, f"grad_dev_{dtype}"])
            limit = NEGATIVE_CONTROL_ULPS * EPS[dtype]
            good = dev <= limit
            passed &= good
            lines.append(f"{'PASS' if good else 'FAIL'}  negative control {name} {dtype}: "
                         f"max deviation {dev:.2e} (limit {limit:.1e} = {NEGATIVE_CONTROL_ULPS:.0e} unit roundoffs)")
        good = summary.loc[name, "classification"] == "scheduling"
        passed &= good
        lines.append(f"{'PASS' if good else 'FAIL'}  negative control {name} classified "
                     f"{summary.loc[name, 'classification']} (must be scheduling)")

    for name, spec in SPECS.items():
        if spec.role != "sensitivity-control":
            continue
        got = summary.loc[name, "classification"]
        # a defect must never be waved through as scheduling; "unresolved" still counts as flagged
        good = got != "scheduling"
        passed &= good
        lines.append(f"{'PASS' if good else 'FAIL'}  sensitivity control {name}: classified '{got}' "
                     f"(float64 deviation {summary.loc[name, 'fwd_dev_float64']:.2e}, "
                     f"float32 {summary.loc[name, 'fwd_dev_float32']:.2e})")
    return passed, lines


def main() -> int:
    frame = pd.read_csv(ROOT / "results" / "equivalence.csv")
    passed, lines = check(frame)
    print("\n".join(lines))
    print("controls passed" if passed else "CONTROLS FAILED: fix the harness before reading any result")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
