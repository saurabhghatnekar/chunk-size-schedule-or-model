"""Decision rule for the kernel tier.

A Triton kernel computes in float32 (or lower) and offers no float64 path, so
the precision ladder is replaced by a float64 oracle and a measured floor:

    floor    = max( quantisation floor of the output format, FLOOR_ULPS32 float32 unit roundoffs )
    d_cross  = worst deviation between chunk lengths, outputs and gradients
    d_oracle = worst deviation from the float64 oracle

    STRUCTURAL    if d_cross  >= STRUCTURAL_X * floor
    ROUNDING      if d_oracle <= ROUNDING_X   * floor
    INCONCLUSIVE  otherwise

These constants were fixed before the kernel tier was run on any GPU.

In low-precision formats the floor is high, so the test is correspondingly blind:
with bfloat16 outputs it cannot see a chunk-length effect much below a few
percent. That is a property of the format the kernel ships in, and the document
reports the floor beside every verdict.
"""
from __future__ import annotations

import math

import pandas as pd

from ..core import EPS
from ..precision_scaling import classify

FLOOR_ULPS32 = 1e3
ROUNDING_X = 20.0
STRUCTURAL_X = 100.0


def kernel_verdict(d_cross: float, d_oracle: float, floor: float) -> str:
    if any(math.isnan(v) for v in (d_cross, d_oracle, floor)):
        return "INCONCLUSIVE"
    if d_cross >= STRUCTURAL_X * floor:
        return "STRUCTURAL"
    if d_oracle <= ROUNDING_X * floor:
        return "ROUNDING"
    return "INCONCLUSIVE"


def _labels(frame: pd.DataFrame) -> str:
    pairs = [(v, c) for v, c in zip(frame["variant"], frame["chunk"]) if isinstance(v, str)]
    return " ".join(dict.fromkeys(f"{v}:{int(c)}" for v, c in pairs))


def summarise_kernels(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (gpu, kernel, input format)."""
    rows = []
    status = frame["status"].astype(str)
    for (gpu, arch, dtype), group in frame.groupby(["gpu", "architecture", "dtype"], sort=False, dropna=False):
        st = status.loc[group.index]
        ran = group[(st == "ok") | st.str.startswith("forward only")]
        full = group[st == "ok"]
        refused = group[st.str.startswith("error")]
        base = {
            "tier": "kernel", "gpu": gpu, "architecture": arch, "role": group["role"].iloc[0],
            "implementation": group["implementation"].iloc[0], "prior": group["prior"].iloc[0],
            "mechanism": group["mechanism"].iloc[0], "dtype": dtype,
            "ran": _labels(ran), "forward_only": _labels(group[st.str.startswith("forward only")]),
            "refused": _labels(refused),
            "refusal": " | ".join(dict.fromkeys(s[len("error: "):][:150] for s in status.loc[refused.index])),
        }
        if ran.empty:
            rows.append({**base, "verdict": "NOT RUN", "classification": "did not run in this format"})
            continue
        floor_measured = max(ran["floor_fwd_max_rel"].max(), full["floor_grad_max_rel"].max() if len(full) else 0.0)
        floor = max(floor_measured, FLOOR_ULPS32 * EPS["float32"])
        # a cross-chunk gradient comparison needs at least two variants with a backward pass
        with_grads = full.groupby("seq_len")["variant"].count().max() if len(full) else 0
        cross_grad = full["grad_max_rel"].max() if with_grads >= 2 else float("nan")
        oracle_grad = full["oracle_grad_max_rel"].max() if len(full) else float("nan")
        d_cross = max(v for v in (ran["fwd_max_rel"].max(), cross_grad) if not math.isnan(v))
        d_oracle = max(v for v in (ran["oracle_fwd_max_rel"].max(), oracle_grad) if not math.isnan(v))
        d_def = ran["definition_fwd_max_rel"].max() if "definition_fwd_max_rel" in ran else 0.0
        d_def = 0.0 if pd.isna(d_def) else float(d_def)
        v = kernel_verdict(d_cross, d_oracle, floor)
        n_variants = ran.groupby("seq_len")["variant"].count().max()
        if n_variants < 2 and v != "STRUCTURAL":
            label = "single variant: matches its reference" if v == "ROUNDING" else "single variant: unresolved"
            v = "ONE VARIANT"  # one variant cannot say anything about chunk length
        else:
            label = classify(v, base["prior"])
        # inputs that get an all-zero gradient in every variant that has a backward pass
        per_row = [set(str(v).split()) for v in full["inputs_without_gradient"].fillna("")] if len(full) else []
        missing = set.intersection(*per_row) if per_row else set()
        rows.append({
            **base,
            "output_dtype": ran["output_dtype"].iloc[0],
            "cross_fwd": ran["fwd_max_rel"].max(), "cross_grad": cross_grad,
            "oracle_fwd": ran["oracle_fwd_max_rel"].max(), "oracle_grad": oracle_grad,
            "definition_cross": d_def,
            "floor": floor, "cross_over_floor": d_cross / floor, "oracle_over_floor": d_oracle / floor,
            "matches_reference": bool(d_oracle <= ROUNDING_X * floor),
            "inputs_without_gradient": " ".join(sorted(missing)),
            "verdict": v, "classification": label,
        })
    return pd.DataFrame(rows)
