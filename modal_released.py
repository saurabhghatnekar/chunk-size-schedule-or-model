"""Run the equivalence test on publicly released checkpoints (src/equivalence/released.py) on Modal.

    uv run modal run modal_released.py                      # Mamba-2 130M, DeltaNet 1.3B, GLA 340M
    uv run modal run modal_released.py --only mamba2

Each model runs in its own container, and the float32 models run twice: once with Triton's default
dot-product precision and once with IEEE. Rows are written to results/equivalence_released.csv as each
container returns.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

FLA = ("https://github.com/fla-org/flash-linear-attention.git", "cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b")
MAMBA = ("https://github.com/state-spaces/mamba.git", "e9594ce1c732d97440f0332fdc43170a2294dbfa")
TP = "/opt/third_party"
GPU = os.environ.get("CHUNK_GPU", "A10G")
TIMEOUT = 2400


def _clone(name: str, url: str, sha: str) -> str:
    return (f"mkdir -p {TP}/{name} && cd {TP}/{name} && git init -q && git remote add origin {url} "
            f"&& git fetch -q --depth 1 origin {sha} && git checkout -q FETCH_HEAD")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "curl")
    .pip_install("torch==2.15.0", "numpy", "pandas", "einops")
    .run_commands(
        _clone("fla", *FLA), f"pip install {TP}/fla",
        "mkdir -p /opt/data && curl -sSL -o /opt/data/enwik8.zip https://mattmahoney.net/dc/enwik8.zip "
        "&& python -c \"import zipfile; zipfile.ZipFile('/opt/data/enwik8.zip').extractall('/opt/data')\" "
        "&& rm /opt/data/enwik8.zip",
    )
    # mamba_ssm is used from source: its Python and Triton code only, not its compiled extensions
    .run_commands(_clone("mamba", *MAMBA), "pip install packaging")
    .env({"CHUNK_THIRD_PARTY": TP, "TORCH_COMPILE_DISABLE": "1"})
    .add_local_python_source("src")
)

app = modal.App("chunk-size-released-checkpoints", image=image)


@app.function(gpu=GPU, cpu=8.0, memory=32768, timeout=TIMEOUT, max_containers=10, single_use_containers=True)
def released_run(name: str, arithmetic: str) -> list[dict]:
    if arithmetic == "ieee":
        os.environ["TRITON_F32_DEFAULT"] = "ieee"  # must be set before Triton is imported
    from src.equivalence.released import RUNNERS

    rows = RUNNERS[name](arithmetic)
    return [{**row, "case": name, "sources": f"fla@{FLA[1][:7]} mamba@{MAMBA[1][:7]}"} for row in rows]


@app.function(gpu=GPU, cpu=8.0, memory=32768, timeout=TIMEOUT, max_containers=10, single_use_containers=True)
def released_control(name: str) -> list[dict]:
    os.environ["TRITON_F32_DEFAULT"] = "ieee"  # the float32 yardstick should be real float32
    from src.equivalence.released import CONTROLS

    rows = CONTROLS[name]("ieee")
    return [{**row, "case": name, "sources": f"fla@{FLA[1][:7]} mamba@{MAMBA[1][:7]}"} for row in rows]


@app.local_entrypoint()
def control(only: str = ""):
    """The bfloat16 control: is changing the chunk size special, or ordinary rounding noise?

        uv run modal run modal_released.py::control
    """
    import datetime as dt

    import pandas as pd

    names = [n for n in ("mamba2", "gla", "delta_net") if not only or n in only.split(",")]
    out = Path(__file__).resolve().parent / "results" / "bf16_control.csv"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(frame):
        frame = frame[~frame["case"].isin(names)]
    for name, call in [(n, released_control.spawn(n)) for n in names]:
        try:
            rows = call.get()
            print(f"  returned {name}: {len(rows)} rows", flush=True)
        except Exception as exc:
            text = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
            print(f"  FAILED {name}: {type(exc).__name__}: {text}", flush=True)
            rows = [{"case": name, "status": f"error: {type(exc).__name__}: {text}"}]
        frame = pd.concat([frame, pd.DataFrame(rows).assign(run_utc=stamp)], ignore_index=True)
        frame.to_csv(out, index=False)
    print(f"wrote {out}")


@app.local_entrypoint()
def main(only: str = ""):
    import datetime as dt

    import pandas as pd

    units = [("mamba2", "tf32 default"), ("mamba2", "ieee"), ("delta_net", "tf32 default"),
             ("gla", "tf32 default"), ("gla", "ieee")]
    units = [u for u in units if not only or u[0] in only.split(",")]
    out = Path(__file__).resolve().parent / "results" / "equivalence_released.csv"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    kept = pd.DataFrame()
    if out.exists():
        previous = pd.read_csv(out)
        redo = set(units)
        kept = previous[[(c, a) not in redo for c, a in zip(previous["case"], previous["arithmetic"])]]
    frame = kept
    calls = [(unit, released_run.spawn(*unit)) for unit in units]
    for unit, call in calls:
        try:
            rows = call.get()
            print(f"  returned {unit}: {len(rows)} rows", flush=True)
        except Exception as exc:
            text = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
            print(f"  FAILED {unit}: {type(exc).__name__}: {text}", flush=True)
            rows = [{"case": unit[0], "arithmetic": unit[1], "status": f"error: {type(exc).__name__}: {text}"}]
        frame = pd.concat([frame, pd.DataFrame(rows).assign(run_utc=stamp)], ignore_index=True)
        frame.to_csv(out, index=False)
    print(f"wrote {out}")
