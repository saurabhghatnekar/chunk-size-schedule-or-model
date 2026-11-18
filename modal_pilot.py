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
LACT = ("https://github.com/a1600012888/LaCT.git", "a648340f9798f173227a0626fde66a6e9b65879a")
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
    .run_commands(
        f"mkdir -p {TP}/lact && cd {TP}/lact && git init -q && git remote add origin {LACT[0]} "
        f"&& git fetch -q --depth 1 origin {LACT[1]} && git checkout -q FETCH_HEAD",
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


@app.function(gpu=GPU, cpu=8.0, timeout=RUN_TIMEOUT, max_containers=10, single_use_containers=True)
def train_arch_run(config: dict, thresholds_sha: str) -> list[dict]:
    from src.pilot.architectural import ArchConfig, run

    rows = run(ArchConfig(**config))
    return [{**row, "thresholds_sha256": thresholds_sha} for row in rows]


@app.function(gpu=GPU, cpu=8.0, timeout=4 * 3600, max_containers=10, single_use_containers=True, retries=1)
def train_arch_long(config: dict, thresholds_sha: str) -> list[dict]:
    """As train_arch_run, for the larger model: a longer cap and one retry if the container is preempted."""
    from src.pilot.architectural import ArchConfig, run

    rows = run(ArchConfig(**config))
    return [{**row, "thresholds_sha256": thresholds_sha} for row in rows]


@app.function(gpu=GPU, cpu=8.0, timeout=1800, max_containers=10, single_use_containers=True)
def bench_run(config: dict, steps: int = 30) -> dict:
    from src.pilot.architectural import ArchConfig, benchmark

    return benchmark(ArchConfig(**config), steps=steps)


SIZES = {  # name: (width, heads, layers); parameter counts are reported by the run itself
    "5M": (256, 4, 6), "42M": (512, 8, 12), "160M": (1024, 16, 12), "410M": (1024, 16, 32),
}


@app.local_entrypoint()
def bench(sizes: str = "42M,160M,410M", chunks: str = "32,128,512,1024", batch: int = 0):
    """Training throughput on random tokens at the planned sizes, for the compute budget.

        uv run modal run modal_pilot.py::bench
        CHUNK_GPU=H100 uv run modal run modal_pilot.py::bench --sizes 160M,410M
    """
    import datetime as dt

    import pandas as pd

    configs = []
    for name in sizes.split(","):
        d, h, layers = SIZES[name]
        size_batch = batch or {"5M": 32, "42M": 16, "160M": 8, "410M": 4}[name]
        configs += [{"chunk": int(c), "seed": 0, "d_model": d, "n_heads": h, "n_layers": layers, "batch": size_batch}
                    for c in chunks.split(",")]
    rows = []
    for config, result in zip(configs, bench_run.map(configs, return_exceptions=True)):
        rows.append(result if isinstance(result, dict) else {**config, "status": f"error: {type(result).__name__}: {str(result)[:150]}"})
    frame = pd.DataFrame(rows)
    frame["size"] = [next(n for n, v in SIZES.items() if v == (r["d_model"], r["n_heads"], r["n_layers"])) for r in rows]
    frame["run_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = Path(__file__).resolve().parent / "results" / "throughput_lact.csv"
    if out.exists():
        previous = pd.read_csv(out)
        key = lambda f: (f["gpu"].astype(str) + "|" + f["size"].astype(str) + "|" + f["chunk"].astype(str)
                         + "|" + f["batch"].astype(str))
        frame = pd.concat([previous[~key(previous).isin(set(key(frame)))], frame], ignore_index=True)
    frame.to_csv(out, index=False)
    cols = ["gpu", "size", "params", "chunk", "batch", "seq_len", "status", "tokens_per_s", "peak_memory_gb"]
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))
    print(f"wrote {out}")


@app.local_entrypoint()
def confirm(rung: str = "A", steps: int = 2000):
    """The confirmatory pilot (docs/go_no_go_confirmatory.md).

        uv run modal run modal_pilot.py::confirm --rung A     # 5.2M parameters, new seeds 5-10, plus reruns
        uv run modal run modal_pilot.py::confirm --rung B     # about 39M parameters, seeds 5-7
    """
    import datetime as dt

    import pandas as pd

    root = Path(__file__).resolve().parent
    sha = hashlib.sha256((root / "docs" / "go_no_go_confirmatory.md").read_bytes()).hexdigest()
    out = root / "results" / "pilot_confirmatory.csv"
    if rung == "A":
        size = {"d_model": 256, "n_heads": 4, "n_layers": 6, "batch": 32}
        configs = [{"chunk": c, "seed": s, "replicate": 0} for c in (32, 512, 1024) for s in range(5, 11)]
        configs += [{"chunk": c, "seed": 5, "replicate": r} for c in (32, 512) for r in (1, 2)]
        function = train_arch_run
    elif rung == "B":
        size = {"d_model": 512, "n_heads": 8, "n_layers": 12, "batch": 16}
        configs = [{"chunk": c, "seed": s, "replicate": 0} for c in (32, 512, 1024) for s in (5, 6, 7)]
        function = train_arch_long
    else:
        raise SystemExit(f"unknown rung {rung}")
    configs = [{**c, **size, "steps": steps} for c in configs]
    key_cols = ["d_model", "chunk", "seed", "replicate"]
    kept = pd.DataFrame()
    if out.exists():  # keep everything already completed, for this rung and the other one
        previous = pd.read_csv(out)
        kept = previous[previous["status"] == "ok"]
        done = set(map(tuple, kept[kept["final"] == True][key_cols].astype(int).values.tolist()))  # noqa: E712
        configs = [c for c in configs if tuple(c[k] for k in key_cols) not in done]
        if not configs:
            raise SystemExit("every configuration of this rung already has a completed run")
    print(f"rung {rung}: {len(configs)} run(s), criteria sha256 {sha[:16]}", flush=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mine = pd.DataFrame()
    for result in function.map(configs, [sha] * len(configs), return_exceptions=True, order_outputs=False):
        if isinstance(result, list):
            rows = result
            last = rows[-1]
            print(f"  returned size={last['d_model']} chunk={last['chunk']} seed={last['seed']} rep={last['replicate']}: "
                  f"{last.get('status')} val {last.get('val_loss', float('nan')):.4f}", flush=True)
        else:
            text = str(result).strip().splitlines()[0][:200] if str(result).strip() else ""
            print(f"  FAILED: {type(result).__name__}: {text}", flush=True)
            rows = [{**size, "status": f"error: {type(result).__name__}: {text}", "final": False, "thresholds_sha256": sha}]
        mine = pd.concat([mine, pd.DataFrame(rows).assign(run_utc=stamp, rung=rung)], ignore_index=True)
        # re-read before every write: the other rung may be writing to the same file from another process
        current = pd.read_csv(out) if out.exists() else pd.DataFrame()
        others = current[current["rung"] != rung] if "rung" in current else current.iloc[:0]
        mine_kept = kept[kept["rung"] == rung] if "rung" in kept else kept.iloc[:0]
        pd.concat([others, mine_kept, mine], ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


@app.local_entrypoint()
def arch(mode: str = "smoke", steps: int = 2000):
    """The architectural-class pilot (docs/go_no_go_architectural.md).

        uv run modal run modal_pilot.py::arch --mode smoke
        uv run modal run modal_pilot.py::arch --mode full
    """
    import datetime as dt

    import pandas as pd

    root = Path(__file__).resolve().parent
    sha = hashlib.sha256((root / "docs" / "go_no_go_architectural.md").read_bytes()).hexdigest()
    out = root / "results" / ("pilot_architectural_smoke.csv" if mode == "smoke" else "pilot_architectural.csv")
    kept = pd.DataFrame()
    if mode == "smoke":
        configs = [{"chunk": c, "seed": 0, "steps": 150, "n_evals": 1} for c in (32, 512, 1024)]
    else:
        configs = [{"chunk": c, "seed": s, "steps": steps} for c in (32, 128, 512, 1024) for s in range(5)]
        if out.exists():  # run only what has no completed final row
            previous = pd.read_csv(out)
            kept = previous[previous["status"] == "ok"]
            done = set(map(tuple, kept[kept["final"] == True][["chunk", "seed"]].values.tolist()))  # noqa: E712
            configs = [c for c in configs if (c["chunk"], c["seed"]) not in done]
            if not configs:
                raise SystemExit("every configuration already has a completed run")
    print(f"{len(configs)} run(s), criteria sha256 {sha[:16]}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame = kept
    for config, result in zip(configs, train_arch_run.map(configs, [sha] * len(configs), return_exceptions=True)):
        if isinstance(result, list):
            rows = result
            print(f"  returned chunk={config['chunk']} seed={config['seed']}: {rows[-1].get('status')} "
                  f"val {rows[-1].get('val_loss', float('nan')):.4f}")
        else:
            text = str(result).strip().splitlines()[0][:200] if str(result).strip() else ""
            print(f"  FAILED chunk={config['chunk']} seed={config['seed']}: {type(result).__name__}: {text}")
            rows = [{**config, "status": f"error: {type(result).__name__}: {text}", "final": False, "thresholds_sha256": sha}]
        frame = pd.concat([frame, pd.DataFrame(rows).assign(run_utc=stamp)], ignore_index=True)
        frame.to_csv(out, index=False)  # after every run, so a lost connection costs nothing already returned
    cols = ["chunk", "seed", "step", "status", "train_loss", "val_loss", "tokens_per_s", "wall_s", "params",
            "fast_weight_updates_per_sequence"]
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))
    print(f"wrote {out}")


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
