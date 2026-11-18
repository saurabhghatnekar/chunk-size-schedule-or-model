"""The precision-scaling diagnostic: separate a numerical artefact from a structural difference.

A deviation caused by re-associated floating-point sums is proportional to the
unit roundoff of the format, so it collapses by many orders of magnitude between
float32 and float64. A deviation caused by a different computation (a load-bearing
chunk boundary, or a bug) is a property of the function and stays where it is.

Decision rule. The thresholds were written down before the full sweep ran, but after a float64
smoke test of the layers had been seen, so they are not blind to the data:

    d32, d64 = worst deviation over chunk lengths (outputs and gradients) in float32, float64
    ROUNDING     if d64 <= TOL64 and d32 / d64 >= SHRINK_MIN       (or both are exactly zero)
    STRUCTURAL   if d64 >  TOL64 and d32 / d64 <= PERSIST_MAX
    INCONCLUSIVE otherwise  -> must be investigated, never silently passed

The class then follows from the verdict and the mathematical prior of the
architecture: ROUNDING -> scheduling; STRUCTURAL with a named boundary
mechanism -> architectural; STRUCTURAL where the chunkwise form is provably
exact -> implementation bug.

    uv run python -m src.equivalence.precision_scaling   # reads results/equivalence.csv
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from .model import ROOT

TOL64 = 1e-9          # "agrees in float64": about 1e7 unit roundoffs of headroom for accumulation
SHRINK_MIN = 1e4      # rounding must shrink at least this much from float32 to float64 (ideal: 2**29 ~ 5e8)
PERSIST_MAX = 1e2     # a structural difference may not shrink by more than this

RESULTS = ROOT / "results"


def verdict(d32: float, d64: float) -> str:
    if d64 == 0.0 and d32 == 0.0:
        return "ROUNDING"
    ratio = d32 / d64 if d64 > 0 else math.inf
    if d64 <= TOL64 and ratio >= SHRINK_MIN:
        return "ROUNDING"
    if d64 > TOL64 and ratio <= PERSIST_MAX:
        return "STRUCTURAL"
    return "INCONCLUSIVE"


def classify(verdict_: str, prior: str) -> str:
    if verdict_ == "ROUNDING":
        return "scheduling"
    if verdict_ == "STRUCTURAL":
        return "architectural" if prior == "boundary" else "implementation bug"
    return "unresolved"


def loglog_slope(eps: list[float], dev: list[float]) -> float:
    """Least-squares slope of log(deviation) against log(unit roundoff): ~1 rounding, ~0 structural."""
    pts = [(math.log10(e), math.log10(d)) for e, d in zip(eps, dev) if d > 0 and math.isfinite(d)]
    if len(pts) < 2:
        return float("nan")
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (tier, architecture): worst deviation per precision, the verdict and the class.

    Forward deviations use every row whose outputs are finite; gradient deviations
    use every row whose gradients are finite. The verdict needs both in float32
    and float64 and is INCONCLUSIVE if either is missing.
    """
    frame = frame[frame["chunk"].notna()].copy()
    rows = []
    for (tier, arch), group in frame.groupby(["tier", "architecture"], sort=False):
        fwd = group[group["status"] != "non-finite output"].groupby("dtype")["fwd_max_rel"].max()
        grad = group[group["status"] == "ok"].groupby("dtype")["grad_max_rel"].max()
        eps = group.groupby("dtype")["unit_roundoff"].first()

        def dev(name):
            if name not in fwd.index or name not in grad.index:
                return float("nan")
            return max(fwd[name], grad[name])

        d32, d64 = dev("float32"), dev("float64")
        v = "INCONCLUSIVE" if math.isnan(d32) or math.isnan(d64) else verdict(d32, d64)
        prior = group["prior"].iloc[0]
        row = {
            "tier": tier,
            "architecture": arch,
            "role": group["role"].iloc[0],
            "implementation": group["implementation"].iloc[0].split(" (")[0],
            "prior": prior,
            "mechanism": group["mechanism"].iloc[0],
            "variants": " ".join(dict.fromkeys(
                f"{v_}:{int(c)}" if tier == "library" else str(int(c))
                for v_, c in zip(group["variant"], group["chunk"]))),
        }
        for name in ["bfloat16", "float16", "float32", "float64"]:
            row[f"fwd_dev_{name}"] = fwd.get(name, float("nan"))
            row[f"grad_dev_{name}"] = grad.get(name, float("nan"))
        row["shrink_f32_to_f64"] = d32 / d64 if d64 > 0 else float("inf")
        ladder = [n for n in fwd.index if n in eps.index]
        row["loglog_slope"] = loglog_slope([eps[n] for n in ladder], [fwd[n] for n in ladder])
        row["non_finite_rows"] = int((group["status"] != "ok").sum())
        row["verdict"] = v
        row["classification"] = classify(v, prior)
        if row["role"] == "implementation-consistency":
            # not a chunk-length sweep: two implementations of one function at one chunk length
            row["classification"] = {"ROUNDING": "implementations agree", "STRUCTURAL": "implementations disagree"}.get(v, "unresolved")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    frame = pd.read_csv(RESULTS / "equivalence.csv")
    summary = summarise(frame)
    out = RESULTS / "precision_scaling.csv"
    summary.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2e}".format):
        print(summary[["tier", "architecture", "fwd_dev_float32", "fwd_dev_float64", "grad_dev_float64",
                       "shrink_f32_to_f64", "loglog_slope", "verdict", "classification"]].to_string(index=False))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
