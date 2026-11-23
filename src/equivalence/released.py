"""The equivalence test on publicly released checkpoints, through the code their authors ship.

Three models, each on a batch of real text (two windows of 1024 tokens from enwik8, tokenised with the
model's own tokenizer):

- `state-spaces/mamba2-130m` through the official `mamba_ssm` kernels, at several chunk sizes;
- `fla-hub/delta_net-1.3B-100B` through flash-linear-attention's DeltaNet kernels: chunked at 16, 32 and
  64 against the recurrent kernel;
- `fla-hub/gla-340M-15B` through the GLA kernels: chunked (size fixed by the kernel) against recurrent.

Two levels:

- model level: the logits of the whole network, compared across chunk sizes and against a reference
  configuration (the recurrent kernel where the library has one, the default chunk size otherwise);
- kernel level: the inputs one layer actually hands to its kernel are captured during a forward pass and
  the kernel is replayed on them at each chunk size, against the float64 token-by-token recurrence.
  This is the kernel tier of the main test with real activations in place of random ones.

Needs a CUDA GPU. `regime` is "fp32" (whatever dot-product precision the container's Triton is set to),
or "bf16" (bfloat16 autocast).
"""
from __future__ import annotations

import functools
import sys
import time

import torch
import torch.nn.functional as F

from .archs.delta import recurrent_gated_delta
from .archs.gla import recurrent_gla
from .archs.mamba2 import recurrent_ssd
from .core import deviation, grad_deviation

SEQ_LEN, WINDOWS = 1024, 2
DATA = "/opt/data/enwik8"


def text_batch(tokenizer) -> torch.Tensor:
    raw = open(DATA, "rb").read()[50_000_000:50_400_000].decode("utf-8", errors="ignore")
    ids = tokenizer(raw, return_tensors="pt").input_ids[0]
    return ids[: WINDOWS * SEQ_LEN].view(WINDOWS, SEQ_LEN).cuda()


def _environment(arithmetic: str) -> dict:
    import triton

    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
            "triton_f32_default": str(triton.knobs.language.fp32_default), "arithmetic": arithmetic}


def _model_rows(base: dict, regime: str, variants: list, run) -> list[dict]:
    """`variants` is a list of (label, chunk, setup); the first one that runs is the reference."""
    rows, ref = [], None
    for label, chunk, setup in variants:
        row = {**base, "level": "model", "regime": regime, "variant": label, "chunk": chunk}
        try:
            setup()
            tic = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=regime == "bf16"):
                out = run().float().cpu()
            torch.cuda.synchronize()
        except Exception as exc:
            text = " / ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())[:300]
            rows.append({**row, "status": f"error: {type(exc).__name__}: {text}"})
            print(f"  model {regime} {label}:{chunk} ERROR {type(exc).__name__}: {text[:120]}", flush=True)
            continue
        if ref is None:
            ref = (label, chunk, out)
        dev = deviation(ref[2], out)
        agree = (out.argmax(-1) == ref[2].argmax(-1)).float().mean().item()
        rows.append({**row, "status": "ok", "ref_variant": ref[0], "ref_chunk": ref[1], "fwd_max_rel": dev["max_rel"],
                     "fwd_l2_rel": dev["l2_rel"], "top1_agreement": agree, "seconds": round(time.perf_counter() - tic, 2)})
        print(f"  model {regime} {label}:{chunk}  dev from {ref[0]}:{ref[1]} {dev['max_rel']:.2e}  top-1 agreement {agree:.6f}", flush=True)
    return rows


def _kernel_rows(base: dict, regime: str, layer: int, leaves: dict, variants: list, oracle) -> list[dict]:
    """Replay one layer's kernel call on its captured inputs. `variants` is a list of (label, chunk, fn),
    `fn(inputs) -> output`; `oracle(inputs in float64) -> output` is the token-by-token recurrence."""
    rows = []
    x64 = {n: t.detach().double().cpu().requires_grad_() for n, t in leaves.items()}
    want = oracle(x64)
    probe = torch.randn(want.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    want_grads = dict(zip(x64, torch.autograd.grad((want * probe).sum(), list(x64.values()), allow_unused=True)))
    want_grads = {n: (g if g is not None else torch.zeros_like(x64[n])) for n, g in want_grads.items()}
    want = want.detach()
    floor = deviation(want, want.to(next(iter(leaves.values())).dtype))["max_rel"]
    for label, chunk, fn in variants:
        row = {**base, "level": "kernel", "regime": regime, "layer": layer, "variant": label, "chunk": chunk}
        try:
            live = {n: t.detach().clone().requires_grad_() for n, t in leaves.items()}
            out = fn(live)
            row.update(status="ok", output_dtype=str(out.dtype).replace("torch.", ""),
                       oracle_fwd_max_rel=deviation(want, out.detach().cpu())["max_rel"], floor_fwd_max_rel=floor)
            try:
                grads = torch.autograd.grad((out.double() * probe.cuda()).sum(), list(live.values()), allow_unused=True)
                grads = {n: (g.detach().cpu() if g is not None else torch.zeros_like(x64[n])) for n, g in zip(live, grads)}
                row["oracle_grad_max_rel"] = grad_deviation(want_grads, grads)["max_rel"]
            except Exception as exc:
                row["status"] = f"forward only: {type(exc).__name__}: {str(exc).strip().splitlines()[0][:120]}"
        except Exception as exc:
            text = " / ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())[:300]
            row["status"] = f"error: {type(exc).__name__}: {text}"
        rows.append(row)
        print(f"  kernel {regime} layer {layer} {label}:{chunk}  {row['status'][:40]}  oracle "
              f"{row.get('oracle_fwd_max_rel', float('nan')):.2e}  grad {row.get('oracle_grad_max_rel', float('nan')):.2e}", flush=True)
    return rows


class Capture:
    """Wrap a kernel function so that the keyword arguments of chosen calls are kept."""

    def __init__(self, fn, names: list):
        self.fn, self.names, self.calls, self.kept, self.on = fn, names, 0, {}, False
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        if self.on:
            if args:
                kwargs = {**dict(zip(self.names, args)), **kwargs}
                args = ()
            self.kept[self.calls] = kwargs
            self.calls += 1
        return self.fn(*args, **kwargs)

    def start(self):
        self.calls, self.kept, self.on = 0, {}, True

    def stop(self) -> dict:
        self.on = False
        picks = sorted({0, self.calls // 2, self.calls - 1})
        return {i: self.kept[i] for i in picks}


# ---- Mamba-2, official kernels ------------------------------------------------------------------

def mamba2(arithmetic: str, repo: str = "state-spaces/mamba2-130m", source: str = "/opt/third_party/mamba") -> list[dict]:
    import json
    import types

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    sys.path.insert(0, source)
    # mamba_ssm's own hub helper imports names from transformers that newer versions may not have; the
    # weights are loaded here directly instead, as a plain state dict
    sys.modules.setdefault("mamba_ssm.utils.hf", types.SimpleNamespace(load_config_hf=None, load_state_dict_hf=None))
    import mamba_ssm.modules.mamba2 as mamba2_module
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    base = {**_environment(arithmetic), "model": repo, "implementation": "mamba_ssm official kernels", "prior": "exact"}
    config = MambaConfig(**json.load(open(hf_hub_download(repo, "config.json"))))
    model = MambaLMHeadModel(config, device="cuda", dtype=torch.float32)
    model.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cuda", weights_only=True))
    model.eval()
    ids = text_batch(AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b"))
    mixers = [layer.mixer for layer in model.backbone.layers]
    for m in mixers:
        m.use_mem_eff_path = False  # the plain path: torch convolution, then the chunked SSD kernel
    base["params"] = sum(p.numel() for p in model.parameters())
    print(f"mamba2: {repo}, {base['params'] / 1e6:.0f}M parameters, {len(mixers)} layers, default chunk {mixers[0].chunk_size}", flush=True)

    def set_chunk(c):
        for m in mixers:
            m.chunk_size = c

    kernel = mamba2_module.mamba_chunk_scan_combined
    capture = Capture(kernel, ["x", "dt", "A", "B", "C", "chunk_size"])
    mamba2_module.mamba_chunk_scan_combined = capture
    chunks = (256, 16, 32, 64, 128, 512, 1024)
    rows = []
    for regime in ("fp32", "bf16"):
        rows += _model_rows(base, regime, [("chunk", c, functools.partial(set_chunk, c)) for c in chunks],
                            lambda: model(ids).logits)
        set_chunk(256)
        capture.start()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=regime == "bf16"):
            model(ids)
        for layer, kw in capture.stop().items():
            fixed = {k: v for k, v in kw.items() if k not in ("x", "dt", "B", "C", "chunk_size")}
            leaves = {n: kw[n] for n in ("x", "dt", "B", "C")}

            def oracle(x, kw=kw):
                heads, groups = x["x"].shape[2], x["B"].shape[2]
                dt = F.softplus(x["dt"] + kw["dt_bias"].detach().double().cpu()) if kw.get("dt_softplus") else x["dt"]
                a = kw["A"].detach().double().cpu() * dt
                b = x["B"].repeat_interleave(heads // groups, dim=2)
                c = x["C"].repeat_interleave(heads // groups, dim=2)
                d = kw["D"].detach().double().cpu()
                skip = x["x"] * (d[:, None] if d.dim() == 1 else d)
                return recurrent_ssd(x["x"] * dt[..., None], a, b, c) + skip

            variants = [("chunk", c, lambda live, c=c, fixed=fixed: kernel(live["x"], live["dt"], fixed["A"], live["B"], live["C"],
                                                                         chunk_size=c, **{k: v for k, v in fixed.items() if k != "A"}))
                        for c in (16, 64, 256, 1024)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=regime == "bf16"):
                rows += _kernel_rows(base, regime, layer, leaves, variants, oracle)
    return rows


# ---- flash-linear-attention models ----------------------------------------------------------------

def _fla_model(repo: str):
    import fla  # noqa: F401  (registers the model classes)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).cuda().eval()
    return model, text_batch(AutoTokenizer.from_pretrained(repo))


def delta_net(arithmetic: str, repo: str = "fla-hub/delta_net-1.3B-100B") -> list[dict]:
    import fla.layers.delta_net as layer_module
    from fla.layers.delta_net import DeltaNet

    base = {**_environment(arithmetic), "model": repo, "implementation": "flash-linear-attention DeltaNet kernels", "prior": "exact"}
    model, ids = _fla_model(repo)
    layers = [m for m in model.modules() if isinstance(m, DeltaNet)]
    base["params"] = sum(p.numel() for p in model.parameters())
    print(f"delta_net: {repo}, {base['params'] / 1e6:.0f}M parameters, {len(layers)} layers, qk_norm {layers[0].qk_norm}", flush=True)
    chunk_kernel, rec_kernel = layer_module.chunk_delta_rule, layer_module.fused_recurrent_delta_rule

    def use(mode, chunk=None):
        for m in layers:
            m.mode = mode
        layer_module.chunk_delta_rule = functools.partial(chunk_kernel, chunk_size=chunk) if chunk else chunk_kernel

    variants = [("fused_recurrent", 1, functools.partial(use, "fused_recurrent"))]
    variants += [("chunk", c, functools.partial(use, "chunk", c)) for c in (16, 32, 64)]
    rows = []
    for regime in ("bf16", "fp32"):  # the chunked kernel refuses float32; that refusal is recorded
        rows += _model_rows(base, regime, variants, lambda: model(ids).logits)
    # kernel level, in bfloat16 (the only format the chunked kernel accepts)
    capture = Capture(rec_kernel, [])
    layer_module.fused_recurrent_delta_rule = capture
    use("fused_recurrent")
    capture.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(ids)
    kept = capture.stop()
    layer_module.fused_recurrent_delta_rule = rec_kernel
    for layer, kw in kept.items():
        leaves = {n: kw[n] for n in ("q", "k", "v", "beta")}
        l2 = bool(kw.get("use_qk_l2norm_in_kernel"))

        def oracle(x, l2=l2):
            norm = (lambda z: z * torch.rsqrt(z.pow(2).sum(-1, keepdim=True) + 1e-6)) if l2 else (lambda z: z)
            t = lambda z: z.transpose(1, 2)
            q, k = norm(x["q"]), norm(x["k"])
            return t(recurrent_gated_delta(t(q) * k.shape[-1] ** -0.5, t(k), t(x["v"]), t(x["beta"]), torch.zeros_like(t(x["beta"]))))

        kernel_variants = [("fused_recurrent", 1, lambda live, l2=l2: rec_kernel(q=live["q"], k=live["k"], v=live["v"], beta=live["beta"], use_qk_l2norm_in_kernel=l2)[0])]
        kernel_variants += [("chunk", c, lambda live, c=c, l2=l2: chunk_kernel(q=live["q"], k=live["k"], v=live["v"], beta=live["beta"],
                                                                           use_qk_l2norm_in_kernel=l2, chunk_size=c)[0]) for c in (16, 32, 64)]
        rows += _kernel_rows(base, "bf16", layer, leaves, kernel_variants, oracle)
    return rows


def gla(arithmetic: str, repo: str = "fla-hub/gla-340M-15B") -> list[dict]:
    import fla.layers.gla as layer_module
    from fla.layers.gla import GatedLinearAttention

    base = {**_environment(arithmetic), "model": repo, "implementation": "flash-linear-attention GLA kernels", "prior": "exact"}
    model, ids = _fla_model(repo)
    layers = [m for m in model.modules() if isinstance(m, GatedLinearAttention)]
    base["params"] = sum(p.numel() for p in model.parameters())
    print(f"gla: {repo}, {base['params'] / 1e6:.0f}M parameters, {len(layers)} layers", flush=True)
    chunk_kernel, rec_kernel = layer_module.chunk_gla, layer_module.fused_recurrent_gla

    def use(mode):
        for m in layers:
            m.mode = mode

    variants = [("fused_recurrent", 1, functools.partial(use, "fused_recurrent")),
                ("chunk (size fixed by the kernel)", 64, functools.partial(use, "chunk"))]
    rows = []
    for regime in ("fp32", "bf16"):
        rows += _model_rows(base, regime, variants, lambda: model(ids).logits)
        capture = Capture(rec_kernel, [])
        layer_module.fused_recurrent_gla = capture
        use("fused_recurrent")
        capture.start()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=regime == "bf16"):
            model(ids)
        kept = capture.stop()
        layer_module.fused_recurrent_gla = rec_kernel
        for layer, kw in kept.items():
            leaves = {"q": kw["q"], "k": kw["k"], "v": kw["v"], "g": kw["gk"]}

            def oracle(x):
                t = lambda z: z.transpose(1, 2)
                return t(recurrent_gla(t(x["q"]) * x["k"].shape[-1] ** -0.5, t(x["k"]), t(x["v"]), t(x["g"])))

            kernel_variants = [
                ("fused_recurrent", 1, lambda live: rec_kernel(q=live["q"], k=live["k"], v=live["v"], gk=live["g"])[0]),
                ("chunk (size fixed by the kernel)", 64, lambda live: chunk_kernel(q=live["q"], k=live["k"], v=live["v"], g=live["g"])[0]),
            ]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=regime == "bf16"):
                rows += _kernel_rows(base, regime, layer, leaves, kernel_variants, oracle)
    return rows


RUNNERS = {"mamba2": mamba2, "delta_net": delta_net, "gla": gla}
