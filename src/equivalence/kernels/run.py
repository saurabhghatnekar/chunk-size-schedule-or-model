"""Run the kernel tier on a CUDA GPU and return rows.

For each kernel, input format and sequence length: run every variant, take the
gradient of a fixed linear probe of the output with respect to every input
through the kernel's own backward pass, and record

    cross-chunk deviation   against the first variant that ran,
    oracle deviation        against the float64 oracle on the same rounded inputs,
    quantisation floor      the oracle rounded to the kernel's output format, against the oracle.

The floor is the smallest oracle deviation any implementation emitting that
format could show, so deviations are judged in units of it.
"""
from __future__ import annotations

import time

import torch

from ..core import EPS, deviation, grad_deviation
from .cases import CASES, KernelCase

SEED = 11


def environment() -> dict:
    import triton
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "torch_compile": "disabled",
    }


def _oracle(case: KernelCase, x64: dict, chunk: int, probe: torch.Tensor):
    leaves = {n: t.clone().requires_grad_() for n, t in x64.items()}
    out = case.oracle(leaves, chunk)
    grads = torch.autograd.grad((out * probe).sum(), list(leaves.values()), allow_unused=True)
    grads = {n: (g if g is not None else torch.zeros_like(leaves[n])) for n, g in zip(leaves, grads)}
    return out.detach(), grads


def _error(exc: Exception) -> str:
    text = " / ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
    return f"{type(exc).__name__}: {text[:400]}"


def base_row(case: KernelCase, dtype_name: str, seq_len: int) -> dict:
    return {
        "tier": "kernel", "architecture": case.name, "role": case.role, "implementation": case.implementation,
        "prior": case.prior, "mechanism": case.mechanism, "dtype": dtype_name,
        "unit_roundoff": EPS[dtype_name], "seq_len": seq_len, "batch": 2,
    }


def iter_unit(case: KernelCase, seq_len: int, dtype_name: str):
    """One (kernel, input format, sequence length) unit. Yields ("start", {...}) before each variant and
    ("row", {...}) after it, so a caller that loses the process mid-variant still has everything before it
    and knows which variant it was in."""
    master = case.make_inputs(torch.Generator().manual_seed(SEED), seq_len)
    dtype = getattr(torch, dtype_name)
    xr = case.cast(master, dtype) if case.cast else {n: t.to(dtype) for n, t in master.items()}
    x64 = {n: t.double() for n, t in xr.items()}  # the exact values the kernel is given
    base = base_row(case, dtype_name, seq_len)
    probe, ref, oracle_cache = None, None, {}
    for variant in case.variants:
        row = {**base, "variant": variant.label, "chunk": variant.chunk}
        yield "start", {"variant": variant.label, "chunk": variant.chunk}
        try:
            leaves = {n: t.cuda().requires_grad_() for n, t in xr.items()}
            tic = time.perf_counter()
            out = variant.run({n: t.clone() for n, t in leaves.items()}, variant.chunk)
            torch.cuda.synchronize()
        except Exception as exc:  # a kernel refusing a format or a chunk size is a finding, not a crash
            yield "row", {**row, "status": f"error: {_error(exc)}"}
            continue
        if probe is None:
            probe = torch.randn(out.shape, generator=torch.Generator().manual_seed(SEED + 1), dtype=torch.float64)
        backward_error = ""
        try:
            grads = torch.autograd.grad((out.double() * probe.cuda()).sum(), list(leaves.values()), allow_unused=True)
            torch.cuda.synchronize()
            grads = {n: (g.detach().cpu() if g is not None else torch.zeros_like(xr[n])) for n, g in zip(leaves, grads)}
        except Exception as exc:  # some kernels are forward-only; keep the forward comparison
            grads, backward_error = None, _error(exc)
        seconds = time.perf_counter() - tic
        out = out.detach().cpu()
        key = variant.chunk if case.oracle_per_chunk else "any"
        if key not in oracle_cache:
            oracle_cache[key] = _oracle(case, x64, variant.chunk, probe)
        o_out, o_grads = oracle_cache[key]
        if ref is None:
            ref = (variant, out, grads, o_out)
        cross_f, orc_f = deviation(ref[1], out), deviation(o_out, out)
        finite = bool(torch.isfinite(out).all())
        result = {
            **row, "output_dtype": str(out.dtype).replace("torch.", ""),
            "ref_variant": ref[0].label, "ref_chunk": ref[0].chunk,
            "fwd_max_rel": cross_f["max_rel"], "fwd_l2_rel": cross_f["l2_rel"],
            "oracle_fwd_max_rel": orc_f["max_rel"], "oracle_fwd_l2_rel": orc_f["l2_rel"],
            "floor_fwd_max_rel": deviation(o_out, o_out.to(out.dtype))["max_rel"],
            # how much the float64 definition itself changes between this chunk length and the reference one
            "definition_fwd_max_rel": deviation(ref[3], o_out)["max_rel"],
            "seconds": round(seconds, 3),
        }
        if grads is None:
            result["status"] = f"forward only: {backward_error}" if finite else "non-finite"
        else:
            finite = finite and all(bool(torch.isfinite(g).all()) for g in grads.values())
            orc_g = grad_deviation(o_grads, grads)
            result.update({
                "status": "ok" if finite else "non-finite",
                "oracle_grad_max_rel": orc_g["max_rel"], "oracle_grad_l2_rel": orc_g["l2_rel"],
                "oracle_grad_worst_tensor": orc_g["worst_tensor"],
                "floor_grad_max_rel": grad_deviation(o_grads, {n: g.to(grads[n].dtype) for n, g in o_grads.items()})["max_rel"],
                "inputs_without_gradient": " ".join(n for n, g in grads.items() if not g.any()),
            })
            if ref[2] is not None:  # cross-chunk gradient comparison needs a reference that has gradients
                cross_g = grad_deviation(ref[2], grads)
                result.update({"grad_max_rel": cross_g["max_rel"], "grad_l2_rel": cross_g["l2_rel"],
                               "grad_worst_tensor": cross_g["worst_tensor"]})
        yield "row", result


def run_case(case: KernelCase, log=print) -> list[dict]:
    rows = []
    for seq_len in case.seq_lens:
        for dtype_name in case.dtypes:
            unit = [payload for kind, payload in iter_unit(case, seq_len, dtype_name) if kind == "row"]
            ran = [r for r in unit if "fwd_max_rel" in r]
            if ran:
                log(f"  {case.name:36s} {dtype_name:8s} T={seq_len:<5d} cross {max(r['fwd_max_rel'] for r in ran):8.2e}  "
                    f"oracle {max(r['oracle_fwd_max_rel'] for r in ran):8.2e}  floor {max(r['floor_fwd_max_rel'] for r in ran):8.2e}"
                    f"  ({len(unit) - len(ran)} refused)")
            else:
                log(f"  {case.name:36s} {dtype_name:8s} T={seq_len:<5d} nothing ran: {unit[-1]['status'][:110]}")
            rows += unit
    return rows


def run_kernels(names: list[str] | None = None, log=print) -> list[dict]:
    """Run the named cases (default: all) on the current CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("the kernel tier needs a CUDA GPU")
    env = environment()
    log(f"kernel tier on {env['gpu']}  torch {env['torch']}  triton {env['triton']}  cuda {env['cuda']}")
    rows = []
    for case in CASES:
        if names is None or case.name in names:
            rows += [{**r, **env} for r in run_case(case, log)]
    return rows
