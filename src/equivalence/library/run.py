"""Run the library-tier cases and return rows in the same schema as the reference tier."""
from __future__ import annotations

import time

import torch

from ..core import EPS, deviation, finite_status, grad_deviation, summary_line
from . import vendor
from .cases import CASES, LibCase

SEED = 7
MODES = [("float32", False, "as shipped"), ("float64", True, "float32 casts lifted to float64")]


def _run_case(case: LibCase, dtype_name: str, lift: bool):
    dtype = getattr(torch, dtype_name)
    inject = case.inject(lift) if case.inject else None
    module = vendor.load(case.repo, case.relpath, lift=lift, inject=inject)
    gen = torch.Generator().manual_seed(SEED)
    master = case.make_inputs(gen)
    probe = None
    results = []
    with vendor.default_dtype(dtype):
        for variant in case.variants:
            leaves = {name: t.to(dtype).requires_grad_() for name, t in master.items()}
            tic = time.perf_counter()
            # the library ops modify some inputs in place, so hand them non-leaf copies
            out = variant.run(module, {name: t.clone() for name, t in leaves.items()}, variant.chunk)
            if probe is None:
                probe = torch.randn(out.shape, generator=torch.Generator().manual_seed(SEED + 1), dtype=torch.float64)
            grads = torch.autograd.grad((out.double() * probe).sum(), list(leaves.values()), allow_unused=True)
            grads = {n: (g if g is not None else torch.zeros_like(leaves[n])) for n, g in zip(leaves, grads)}
            results.append((variant, out.detach(), grads, str(out.dtype).replace("torch.", ""), time.perf_counter() - tic))
    return results


def run_library(log=print) -> list[dict]:
    rows = []
    for case in CASES:
        for dtype_name, lift, note in MODES:
            base = {
                "tier": "library", "architecture": case.name, "role": case.role,
                "implementation": f"{case.implementation} ({note})", "prior": case.prior,
                "mechanism": case.mechanism, "dtype": dtype_name, "unit_roundoff": EPS[dtype_name],
                "seq_len": 256, "batch": 2,
            }
            try:
                results = _run_case(case, dtype_name, lift)
            except Exception as exc:  # a library failure is a finding to record, not a crash of the sweep
                log(f"  {case.name:32s} {dtype_name:8s} FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
                rows.append({**base, "status": f"error: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"})
                continue
            ref_variant, ref_out, ref_grads, _, _ = results[0]
            for variant, out, grads, out_dtype, seconds in results:
                fwd, bwd = deviation(ref_out, out), grad_deviation(ref_grads, grads)
                rows.append({
                    **base, "chunk": variant.chunk, "variant": variant.label,
                    "ref_chunk": ref_variant.chunk, "ref_variant": ref_variant.label,
                    "status": finite_status(out, grads, ref_out, ref_grads), "output_dtype": out_dtype,
                    "fwd_max_abs": fwd["max_abs"], "fwd_max_rel": fwd["max_rel"], "fwd_l2_rel": fwd["l2_rel"],
                    "grad_max_rel": bwd["max_rel"], "grad_l2_rel": bwd["l2_rel"],
                    "grad_max_rel_per_tensor": bwd["max_rel_per_tensor"], "grad_worst_tensor": bwd["worst_tensor"],
                    "seconds": round(seconds, 3),
                })
            log(summary_line(case.name, dtype_name, rows[-len(results):]))
    return rows
