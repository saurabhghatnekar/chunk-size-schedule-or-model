"""The equivalence test on trained weights.

The classification in docs/classification.md was made on randomly initialised models. Here the same
question is asked of the two pilot models after training:

- reference ladder: the trained weights are run with this repository's reference ops in float32 and
  float64 on CPU, at several chunk lengths, and the precision-scaling rule is applied as before;
- kernels: for the Gated DeltaNet model, the library's Triton kernel on the GPU is compared with a
  float64 recurrence on the same trained weights; for the LaCT model, the official op under its own
  autocast is compared with the float64 reference at the same chunk length.

Outputs are the logits; gradients are those of the next-byte loss with respect to every weight.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ..equivalence.core import deviation, grad_deviation
from .precision import load_data, windows

N_WINDOWS = 2


def _batch(cfg) -> torch.Tensor:
    _, val = load_data(cfg.data_path)
    starts = torch.linspace(1000, val.numel() - cfg.seq_len - 2, N_WINDOWS).long().cuda()
    return windows(val, starts, cfg.seq_len)


def _forward_backward(model, call, tokens):
    model.zero_grad(set_to_none=True)
    logits = call(tokens[:, :-1])
    loss = F.cross_entropy(logits.double().flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    grads = {n: (p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p).cpu())
             for n, p in model.named_parameters()}
    return logits.detach().cpu(), grads


def _ladder(model, tokens, chunks, call_at, base: dict) -> list[dict]:
    """Reference ops on CPU in float32 and float64; deviation of every chunk length from the first."""
    rows = []
    for dtype_name in ("float32", "float64"):
        m = copy.deepcopy(model).cpu().to(getattr(torch, dtype_name))
        tok = tokens.cpu()
        ref = None
        for chunk in chunks:
            out, grads = _forward_backward(m, lambda t, c=chunk: call_at(m, t, c), tok)
            if ref is None:
                ref = (out, grads)
            fwd, bwd = deviation(ref[0], out), grad_deviation(ref[1], grads)
            rows.append({**base, "tier": "trained-reference", "dtype": dtype_name, "chunk": chunk, "ref_chunk": chunks[0],
                         "fwd_max_rel": fwd["max_rel"], "grad_max_rel": bwd["max_rel"], "status": "ok"})
    return rows


def gdn_oracle(model, cfg):
    """The validation batch and the float64 token-by-token recurrence on the trained weights (GPU container)."""
    tokens = _batch(cfg).cpu()
    oracle_model = copy.deepcopy(model).cpu().double()
    oracle = _forward_backward(oracle_model, lambda t: oracle_model(t, 1, cfg.regime, reference=True, mixer="reference"), tokens)
    return tokens, oracle


def gdn_ladder(model, cfg, tokens, base: dict) -> list[dict]:
    """Reference ops in float32 and float64 at several chunk lengths. CPU only; runs anywhere."""
    return _ladder(model, tokens, (1, 16, 64, 256, 1024, 48), lambda m, t, c: m(t, c, cfg.regime, mixer="reference"), base)


def lact_ladder(model, tokens, base: dict) -> list[dict]:
    return _ladder(model, tokens, (32, 128, 512, 1024), lambda m, t, c: m(t, c, reference=True), base)


def gdn_kernel_rows(model, cfg, tokens, oracle, regimes, base: dict) -> list[dict]:
    """The library kernel on the GPU with the trained weights, against the float64 recurrence."""
    rows = []
    model = model.cuda().float()
    tok = tokens.cuda()
    for regime in regimes:
        ref = None
        for chunk in (16, 32, 64):
            try:
                out, grads = _forward_backward(model, lambda t, c=chunk: model(t, c, regime), tok)
            except Exception as exc:
                rows.append({**base, "tier": "trained-kernel", "dtype": regime, "chunk": chunk,
                             "status": f"error: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"})
                continue
            if ref is None:
                ref = (out, grads, chunk)
            cross_f, cross_g = deviation(ref[0], out), grad_deviation(ref[1], grads)
            orc_f, orc_g = deviation(oracle[0], out), grad_deviation(oracle[1], grads)
            rows.append({**base, "tier": "trained-kernel", "dtype": regime, "chunk": chunk, "ref_chunk": ref[2],
                         "fwd_max_rel": cross_f["max_rel"], "grad_max_rel": cross_g["max_rel"],
                         "oracle_fwd_max_rel": orc_f["max_rel"], "oracle_grad_max_rel": orc_g["max_rel"], "status": "ok"})
    return rows


def lact_official_rows(model, cfg, base: dict):
    """The official op under its own bfloat16 autocast, against the float64 reference at the same chunk length."""
    tokens = _batch(cfg)
    chunks = (32, 128, 512, 1024)
    rows = []
    oracle_model = copy.deepcopy(model).cpu().double()
    gpu_model = model.cuda().float()
    ref = None
    for chunk in chunks:
        with torch.no_grad():
            want = oracle_model(tokens.cpu()[:, :-1], chunk, reference=True)
            got = gpu_model(tokens.cuda()[:, :-1], chunk).cpu()
        if ref is None:
            ref = got
        rows.append({**base, "tier": "trained-kernel", "dtype": "bf16 autocast (official op)", "chunk": chunk,
                     "ref_chunk": chunks[0], "fwd_max_rel": deviation(ref, got)["max_rel"],
                     "oracle_fwd_max_rel": deviation(want, got)["max_rel"],
                     "floor_fwd_max_rel": deviation(want, want.to(torch.bfloat16))["max_rel"], "status": "ok"})
    return rows, tokens.cpu()


def main() -> int:
    """The CPU half: reference ladders on the saved trained weights, appended to results/equivalence_trained.csv.

        uv run python -m src.pilot.trained_check
    """
    from pathlib import Path

    import pandas as pd

    from .architectural import ArchConfig, LaCTByteLM
    from .precision import ByteLM, PilotConfig

    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "equivalence_trained.csv"
    frame = pd.read_csv(out)
    frame = frame[frame["tier"] != "trained-reference"]
    rows = []
    for name, build, ladder in (
        ("gated_deltanet", lambda: ByteLM(PilotConfig(regime="bf16", chunk=64, seed=0)),
         lambda m, tok, base: gdn_ladder(m, PilotConfig(regime="bf16", chunk=64, seed=0), tok, base)),
        ("lact", lambda: LaCTByteLM(ArchConfig(chunk=128, seed=0)), lact_ladder),
    ):
        blob = torch.load(root / "checkpoints" / f"trained_{name}.pt", weights_only=True)
        model = build()
        model.load_state_dict(blob["state_dict"])
        kernel = frame[frame["architecture"].str.startswith(name)].iloc[0]
        base = {k: kernel[k] for k in ("architecture", "prior", "gpu", "steps", "trained_val_loss")}
        print(f"reference ladder on trained {name} ...", flush=True)
        rows += ladder(model, blob["tokens"], base)
    full = pd.concat([frame, pd.DataFrame(rows).assign(run_utc=frame["run_utc"].iloc[0])], ignore_index=True)
    full.to_csv(out, index=False)
    print(full[full["tier"] == "trained-reference"][["architecture", "dtype", "chunk", "fwd_max_rel", "grad_max_rel"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
