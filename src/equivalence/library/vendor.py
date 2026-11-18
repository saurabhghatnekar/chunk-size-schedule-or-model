"""Pinned third-party sources, fetched into third_party/ (git-ignored) and loaded by file path.

Loading by path matters: the library packages import Triton at package import
time, which does not exist on a CPU-only machine, while the pure-PyTorch
reference ops inside them need nothing but torch and einops.

`lift=True` returns a copy of the same source with every hard-coded float32 cast
replaced by float64. Several library reference ops cast their inputs to float32
internally, which would make the precision-scaling diagnostic impossible; the
lifted copy runs the identical algorithm in double precision and is used only
for that diagnostic. The "as shipped" rows always come from the unmodified source.
"""
from __future__ import annotations

import os
import re
import subprocess
import types
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
# on the GPU image the pinned sources are baked in at a fixed path instead of being cloned at run time
THIRD_PARTY = Path(os.environ.get("CHUNK_THIRD_PARTY", ROOT / "third_party"))

PINS = {
    "fla": ("https://github.com/fla-org/flash-linear-attention.git", "cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b"),
    "lact": ("https://github.com/a1600012888/LaCT.git", "a648340f9798f173227a0626fde66a6e9b65879a"),
}


def ensure(name: str) -> Path:
    """Clone `name` at its pinned commit if it is not already there; return its directory."""
    url, sha = PINS[name]
    dest = THIRD_PARTY / name
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", url],
                    ["git", "fetch", "-q", "--depth", "1", "origin", sha], ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=dest, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, check=True, capture_output=True, text=True).stdout.strip()
    if head != sha:
        raise RuntimeError(f"third_party/{name} is at {head}, expected pinned {sha}")
    return dest


_LIFT = [
    (re.compile(r"torch\.float32"), "torch.float64"),
    (re.compile(r"\.float\(\)"), ".double()"),
    (re.compile(r"dtype=torch\.float(?![0-9a-z_])"), "dtype=torch.float64"),
]
_STRIP = re.compile(r"^@torch\.(compile|autocast)\(.*\)\s*$", re.MULTILINE)
_FLA_IMPORT = re.compile(r"^from fla\.[\w.]+ import .*$", re.MULTILINE)


def load(repo: str, relpath: str, lift: bool = False, inject: dict | None = None,
         strip: bool = True) -> types.ModuleType:
    """Execute one source file of a pinned repository as a stand-alone module.

    Decorators that need a compiler or a GPU (`torch.compile`, `torch.autocast`)
    are stripped unless `strip=False` (the GPU tier keeps them, so the op runs under
    its own autocast), and intra-package imports are replaced by `inject`.
    """
    source = (ensure(repo) / relpath).read_text()
    if strip:
        source = _STRIP.sub("", source)
    source = _FLA_IMPORT.sub("", source)
    if lift:
        for pattern, replacement in _LIFT:
            source = pattern.sub(replacement, source)
    module = types.ModuleType(f"vendored_{repo}_{Path(relpath).stem}{'_f64' if lift else ''}")
    module.__dict__.update(inject or {})
    exec(compile(source, f"{repo}/{relpath}", "exec"), module.__dict__)
    return module


@contextmanager
def default_dtype(dtype: torch.dtype):
    """Tensors the library creates without an explicit dtype follow the run's precision."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def short(name: str) -> str:
    return f"{name}@{PINS[name][1][:7]}"
