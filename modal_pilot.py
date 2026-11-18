"""Run the small precision pilot (docs/go_no_go.md, section 1) on Modal.

    uv run modal run modal_pilot.py --mode smoke          # three short runs: timing and arithmetic check
    uv run modal run modal_pilot.py --mode full           # 3 arithmetics x 2 chunk lengths x 3 seeds, plus 2 repeats

Rows are written to results/pilot_precision.csv. Each row carries the sha256 of docs/go_no_go.md as it stood
when the pilot was launched.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import modal

FLA = ("https://github.com/fla-org/flash-linear-attention.git", "cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b")
TP = "/opt/third_party"
GPU = os.environ.get("CHUNK_GPU", "A10G")
RUN_TIMEOUT = 2700  # the IEEE arm runs at about a third of the TF32 speed and compiles for several minutes

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "curl")
    .pip_install("torch==2.15.0", "numpy", "pandas", "einops")
    .run_commands(
        f"mkdir -p {TP}/fla && cd {TP}/fla && git init -q && git remote add origin {FLA[0]} "
        f"&& git fetch -q --depth 1 origin {FLA[1]} && git checkout -q FETCH_HEAD",
        f"pip install {TP}/fla",
        "mkdir -p /opt/data && curl -sSL -o /opt/data/enwik8.zip https://mattmahoney.net/dc/enwik8.zip "
        "&& python -c \"import zipfile; zipfile.ZipFile('/opt/data/enwik8.zip').extractall('/opt/data')\" "
        "&& rm /opt/data/enwik8.zip && ls -la /opt/data",
    )
    .env({"CHUNK_THIRD_PARTY": TP, "TORCH_COMPILE_DISABLE": "1"})
    .add_local_python_source("src")
)

app = modal.App("chunk-size-precision-pilot", image=image)


# One run per container. A warm container reused for a different arithmetic would carry kernels compiled under
# the previous one (Triton refuses to run them once the precision setting changes, which is how this was found).
@app.function(gpu=GPU, cpu=8.0, timeout=RUN_TIMEOUT, max_containers=10, single_use_containers=True)
def train_run(config: dict, thresholds_sha: str) -> list[dict]:
    if config["regime"] == "fp32_ieee":
        os.environ["TRITON_F32_DEFAULT"] = "ieee"  # must be set before Triton is imported
    else:
        os.environ.pop("TRITON_F32_DEFAULT", None)
    from src.pilot.precision import PilotConfig, run

    rows = run(PilotConfig(**config))
    return [{**row, "thresholds_sha256": thresholds_sha} for row in rows]


@app.local_entrypoint()
def main(mode: str = "smoke", steps: int = 2000):
    import datetime as dt

    import pandas as pd

    root = Path(__file__).resolve().parent
    sha = hashlib.sha256((root / "docs" / "go_no_go.md").read_bytes()).hexdigest()
    if mode == "smoke":
        configs = [{"regime": r, "chunk": 64, "seed": 0, "steps": 150, "n_evals": 1}
                   for r in ("bf16", "fp32_tf32", "fp32_ieee")]
    elif mode == "full":
        configs = [{"regime": r, "chunk": c, "seed": s, "steps": steps}
                   for r in ("bf16", "fp32_tf32", "fp32_ieee") for c in (16, 64) for s in (0, 1, 2)]
        configs += [{"regime": r, "chunk": 64, "seed": 0, "replicate": 1, "steps": steps} for r in ("bf16", "fp32_tf32")]
    else:
        raise SystemExit(f"unknown mode {mode}")
    out = root / "results" / ("pilot_precision_smoke.csv" if mode == "smoke" else "pilot_precision.csv")
    kept = pd.DataFrame()
    if mode == "full" and out.exists():  # rerun only the configurations that have no completed final row
        previous = pd.read_csv(out)
        previous["replicate"] = previous["replicate"].fillna(0).astype(int) if "replicate" in previous else 0
        kept = previous[previous["status"] == "ok"]
        done = set(map(tuple, kept[kept["final"] == True][["regime", "chunk", "seed", "replicate"]].values.tolist()))  # noqa: E712
        configs = [c for c in configs if (c["regime"], c["chunk"], c["seed"], c.get("replicate", 0)) not in done]
        if not configs:
            raise SystemExit("every configuration already has a completed run")
    print(f"{len(configs)} run(s), thresholds sha256 {sha[:16]}")
    # Write after every run that returns: a client-side failure (for example losing the connection
    # mid-flight) must not cost the runs that already finished.
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame = kept
    for result in train_run.map(configs, [sha] * len(configs), return_exceptions=True, order_outputs=False):
        if isinstance(result, list):
            rows = result
            print(f"  returned {rows[0]['regime']} chunk={rows[0]['chunk']} seed={rows[0]['seed']} rep={rows[0]['replicate']}")
        else:
            text = str(result).strip().splitlines()[0][:200] if str(result).strip() else ""
            print(f"  FAILED: {type(result).__name__}: {text}")
            rows = [{"status": f"error: {type(result).__name__}: {text}", "thresholds_sha256": sha}]
        frame = pd.concat([frame, pd.DataFrame(rows).assign(run_utc=stamp)], ignore_index=True)
        frame.to_csv(out, index=False)
    cols = ["regime", "chunk", "seed", "step", "status", "train_loss", "val_loss_own", "val_loss_ref",
            "tokens_per_s", "wall_s", "kernel_oracle_dev", "triton_f32_default", "solve_tril_precision", "params"]
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))
    frame.to_csv(out, index=False)
    print(f"wrote {out}")
