"""Run the kernel tier of the equivalence test on a Modal GPU.

    uv run modal run modal_app.py                 # every case, default GPU (A10G)
    CHUNK_GPU=H100 uv run modal run modal_app.py  # kernels take different code paths on Hopper
    uv run modal run modal_app.py --only fla_chunk_gla,fla_chunk_simple_gla   # rerun some cases

The image pins flash-linear-attention and LaCT to the commits in
src/equivalence/library/vendor.py. Rows come back to this machine and are merged
into results/equivalence_kernels.csv, keyed by GPU model, so several GPUs can
sit side by side. Nothing is trained; a run is a few minutes of one GPU.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

FLA = ("https://github.com/fla-org/flash-linear-attention.git", "cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b")
LACT = ("https://github.com/a1600012888/LaCT.git", "a648340f9798f173227a0626fde66a6e9b65879a")
TP = "/opt/third_party"


def _clone(name: str, url: str, sha: str) -> str:
    return (f"mkdir -p {TP}/{name} && cd {TP}/{name} && git init -q && git remote add origin {url} "
            f"&& git fetch -q --depth 1 origin {sha} && git checkout -q FETCH_HEAD")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.15.0", "numpy", "pandas", "einops")
    .run_commands(_clone("fla", *FLA), _clone("lact", *LACT), f"pip install {TP}/fla")
    .env({"CHUNK_THIRD_PARTY": TP, "TORCH_COMPILE_DISABLE": "1"})
    .add_local_python_source("src")
)

app = modal.App("chunk-size-equivalence-kernels", image=image)
GPU = os.environ.get("CHUNK_GPU", "A10G")
# Ablation: CHUNK_TRITON_F32=ieee asks Triton for IEEE float32 dot products instead of its TF32 default.
# Rows then go to results/equivalence_kernels_ieee.csv.
TRITON_F32 = os.environ.get("CHUNK_TRITON_F32", "")


# Seconds for one (kernel, input format, sequence length). Compiling and autotuning a kernel for a new block
# size is CPU-bound and slow, so the container gets real cores; with Modal's default fractional CPU the same
# compiles ran several times longer and some units sat compiling until the cap.
UNIT_TIMEOUT = 900


@app.function(gpu=GPU, cpu=8.0, timeout=UNIT_TIMEOUT, max_containers=10)
def kernel_unit(name: str, dtype_name: str, seq_len: int, triton_f32: str = ""):
    """One (kernel, input format, sequence length) per container, streamed back row by row, so a hang in one
    kernel variant costs that variant and nothing else."""
    if triton_f32:
        os.environ["TRITON_F32_DEFAULT"] = triton_f32  # must be set before Triton is imported
    from src.equivalence.kernels.cases import CASES
    from src.equivalence.kernels.run import environment, iter_unit
    from src.equivalence.library import vendor

    assert vendor.PINS["fla"][1] == FLA[1] and vendor.PINS["lact"][1] == LACT[1], "image pins differ from vendor.py"
    # if a variant goes quiet, say where: a Python stack every few minutes distinguishes a kernel that is
    # still being compiled from one that is stuck executing
    import faulthandler
    import sys
    faulthandler.dump_traceback_later(240, repeat=True, file=sys.stderr)
    env = environment()
    if triton_f32:
        import triton
        env["triton_f32_default"] = str(triton.knobs.language.fp32_default)
    yield "env", env
    case = next(c for c in CASES if c.name == name)
    for kind, payload in iter_unit(case, seq_len, dtype_name):
        print(f"{kind:5s} {name} {dtype_name} T={seq_len} {payload.get('variant')}:{payload.get('chunk')} "
              f"{str(payload.get('status', ''))[:90]}", flush=True)
        yield kind, payload


@app.function(gpu=GPU, timeout=600)
def triton_dot_precision() -> str:
    """Evidence for how `tl.dot` treats float32 operands on this GPU: Triton's own documentation of the
    default, and which fla kernels opt in to or out of TF32.  `modal run modal_app.py::triton_dot_precision`"""
    import inspect
    import subprocess

    import torch
    import triton
    import triton.language as tl

    doc = inspect.getdoc(tl.dot) or ""
    lines = [f"GPU {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}, triton {triton.__version__}",
             f"tl.dot signature: {inspect.signature(getattr(tl.dot, '__wrapped__', tl.dot))}", "tl.dot doc:", doc, ""]
    knob = subprocess.run("grep -rn 'F32_DEFAULT\\|fp32_default' /usr/local/lib/python3.12/site-packages/triton/knobs.py "
                          "/usr/local/lib/python3.12/site-packages/triton/language/*.py | head -12",
                          shell=True, capture_output=True, text=True).stdout
    lines += ["Triton float32 dot default knob:", knob]
    import fla.utils as fu
    lines.append(f"fla IS_TF32_SUPPORTED={fu.IS_TF32_SUPPORTED}  IS_NVIDIA_HOPPER={getattr(fu, 'IS_NVIDIA_HOPPER', None)}")
    for sub in ("gated_delta_rule", "delta_rule", "simple_gla", "gla", "common", "utils"):
        out = subprocess.run(f"grep -rn 'tl.dot' {TP}/fla/fla/ops/{sub}/*.py | wc -l; "
                             f"grep -rn 'allow_tf32=False\\|ieee' {TP}/fla/fla/ops/{sub}/*.py | wc -l; "
                             f"grep -rn \"input_precision\\|'tf32'\" {TP}/fla/fla/ops/{sub}/*.py | wc -l",
                             shell=True, capture_output=True, text=True).stdout.split()
        lines.append(f"fla/ops/{sub}: tl.dot calls {out[0]}, lines forcing IEEE {out[1]}, lines naming input_precision/tf32 {out[2]}")
    text = "\n".join(lines)
    print(text)
    return text


@app.local_entrypoint()
def main(only: str = "", dtype: str = ""):
    import datetime as dt
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    from src.equivalence.kernels.cases import CASES
    from src.equivalence.kernels.run import base_row

    cases = [c for c in CASES if not only or c.name in only.split(",")]
    units = [(c, d, t) for c in cases for t in c.seq_lens for d in c.dtypes if not dtype or d in dtype.split(",")]

    def collect(unit):
        case, dtype_name, seq_len = unit
        env, rows, pending = {}, [], None
        try:
            for kind, payload in kernel_unit.remote_gen(case.name, dtype_name, seq_len, TRITON_F32):
                if kind == "env":
                    env = payload
                elif kind == "start":
                    pending = payload
                else:
                    rows.append(payload)
                    pending = None
        except Exception as exc:  # timeout or container crash: keep what arrived, blame the variant in flight
            text = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else ""
            reason = f"did not return within {UNIT_TIMEOUT}s or the container died ({type(exc).__name__}: {text})"
            rows.append({**base_row(case, dtype_name, seq_len), **(pending or {}), "status": f"error: {reason}"})
        print(f"  done {case.name} {dtype_name} T={seq_len}: {len(rows)} row(s)", flush=True)
        return env, rows

    with ThreadPoolExecutor(max_workers=len(units)) as pool:
        results = list(pool.map(collect, units))
    env = next((e for e, _ in results if e), {"gpu": GPU})
    frame = pd.DataFrame([{**row, **env} for _, rows in results for row in rows])
    frame["run_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stem = "equivalence_kernels" + (f"_{TRITON_F32}" if TRITON_F32 else "")
    out = Path(__file__).resolve().parent / "results" / f"{stem}.csv"
    if out.exists():
        previous = pd.read_csv(out)
        same = previous["gpu"].isin(frame["gpu"].unique()) & previous["architecture"].isin([c.name for c in cases])
        frame = pd.concat([previous[~same], frame], ignore_index=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}  ({len(frame)} rows in file, {len(units)} unit(s) run, GPU {env['gpu']})")
