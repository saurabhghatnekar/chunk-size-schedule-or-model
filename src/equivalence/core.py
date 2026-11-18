"""Deviation metrics and the sweep over chunk lengths and precisions."""
from __future__ import annotations

import time

import torch

from .archs import ArchSpec
from .model import ModelConfig, build_model, forward_backward, load_batch, load_checkpoint

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
# unit roundoff of each format (bfloat16 keeps 8 significand bits, float16 keeps 11)
EPS = {"bfloat16": 2.0 ** -8, "float16": 2.0 ** -11, "float32": 2.0 ** -24, "float64": 2.0 ** -53}


def deviation(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    ref, test = ref.double(), test.double()
    diff = (test - ref).abs()
    scale = ref.abs().max().item()
    norm = ref.norm().item()
    return {
        "max_abs": diff.max().item(),
        "max_rel": diff.max().item() / scale if scale > 0 else float("nan"),
        "l2_rel": (test - ref).norm().item() / norm if norm > 0 else float("nan"),
    }


def grad_deviation(ref: dict, test: dict) -> dict[str, float]:
    """Global and worst-single-tensor deviation between two gradient dictionaries."""
    flat_ref = torch.cat([g.double().flatten() for g in ref.values()])
    flat_test = torch.cat([test[n].double().flatten() for n in ref])
    out = deviation(flat_ref, flat_test)
    global_scale = flat_ref.abs().max().item()
    worst, worst_name = 0.0, ""
    for name, g in ref.items():
        diff = (test[name].double() - g.double()).abs().max().item()
        scale = g.double().abs().max().item()
        rel = diff / (scale if scale > 0 else global_scale)
        if rel > worst:
            worst, worst_name = rel, name
    out["max_rel_per_tensor"] = worst
    out["worst_tensor"] = worst_name
    return out


def finite_status(out, grads, ref_out, ref_grads) -> str:
    """'ok', or which half of the comparison is unusable because something overflowed."""
    if not (bool(torch.isfinite(out).all()) and bool(torch.isfinite(ref_out).all())):
        return "non-finite output"
    for g in (*grads.values(), *ref_grads.values()):
        if not bool(torch.isfinite(g).all()):
            return "non-finite gradient"
    return "ok"


def summary_line(name: str, dtype_name: str, rows: list[dict]) -> str:
    fwd = [r["fwd_max_rel"] for r in rows if r["status"] != "non-finite output"]
    grad = [r["grad_max_rel"] for r in rows if r["status"] == "ok"]
    f = f"{max(fwd):9.3e}" if fwd else "non-finite"
    g = f"{max(grad):9.3e}" if grad else "non-finite"
    return f"  {name:32s} {dtype_name:9s} max fwd dev {f:>10s}   max grad dev {g:>10s}"


def run_arch(spec: ArchSpec, cfg: ModelConfig, dtype_names: list[str], log=print) -> list[dict]:
    """Forward and backward at every chunk length and precision; one row per (dtype, chunk)."""
    state, ckpt_sha = load_checkpoint(spec, cfg)
    tokens, batch_sha = load_batch(cfg)
    ref_chunk = spec.chunk_lengths[0]
    rows = []
    for dtype_name in dtype_names:
        base = {
            "tier": "reference",
            "architecture": spec.name,
            "role": spec.role,
            "implementation": "src/equivalence/archs (this repository)",
            "prior": spec.prior,
            "mechanism": spec.mechanism,
            "variant": "chunkwise",
            "ref_variant": "chunkwise",
            "dtype": dtype_name,
            "unit_roundoff": EPS[dtype_name],
            "ref_chunk": ref_chunk,
            "seq_len": cfg.seq_len,
            "batch": cfg.batch,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "checkpoint_sha256": ckpt_sha[:16],
            "batch_sha256": batch_sha[:16],
        }
        try:
            model = build_model(spec, cfg, state, DTYPES[dtype_name])
            results = {}
            for chunk in spec.chunk_lengths:
                tic = time.perf_counter()
                results[chunk] = forward_backward(model, tokens, chunk)
                results[chunk] += (time.perf_counter() - tic,)
        except (RuntimeError, NotImplementedError) as exc:
            log(f"  {spec.name:22s} {dtype_name:9s} unsupported on this backend: {str(exc).splitlines()[0][:90]}")
            rows.append({**base, "chunk": None, "status": f"unsupported: {str(exc).splitlines()[0][:120]}"})
            continue
        ref_logits, _, ref_grads, _ = results[ref_chunk]
        for chunk in spec.chunk_lengths:
            logits, loss, grads, seconds = results[chunk]
            fwd = deviation(ref_logits, logits)
            bwd = grad_deviation(ref_grads, grads)
            rows.append({
                **base,
                "chunk": chunk,
                "status": finite_status(logits, grads, ref_logits, ref_grads),
                "output_dtype": dtype_name,
                "loss": loss,
                "fwd_max_abs": fwd["max_abs"],
                "fwd_max_rel": fwd["max_rel"],
                "fwd_l2_rel": fwd["l2_rel"],
                "grad_max_rel": bwd["max_rel"],
                "grad_l2_rel": bwd["l2_rel"],
                "grad_max_rel_per_tensor": bwd["max_rel_per_tensor"],
                "grad_worst_tensor": bwd["worst_tensor"],
                "seconds": round(seconds, 3),
            })
        log(summary_line(spec.name, dtype_name, rows[-len(spec.chunk_lengths):]))
    return rows
