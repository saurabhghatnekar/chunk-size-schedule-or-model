"""Small precision pilot on the scheduling class (docs/go_no_go.md, section 1).

A byte-level language model whose mixing layer calls flash-linear-attention's
`chunk_gated_delta_rule` Triton kernel directly. A run is fixed by (arithmetic,
chunk length, seed); the seed fixes the initial weights and the order of training
windows, so two runs with the same seed differ only in the two factors under study.

Needs a CUDA GPU. The arithmetic of the Triton kernels is chosen by the caller
*before* Triton is imported (TRITON_F32_DEFAULT); `configure_arithmetic` handles
the rest and reports what is actually in effect.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REGIMES = ("bf16", "fp32_tf32", "fp32_ieee")


@dataclass(frozen=True)
class PilotConfig:
    regime: str
    chunk: int
    seed: int
    replicate: int = 0
    steps: int = 2000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    seq_len: int = 1024
    batch: int = 32
    lr: float = 1e-3
    warmup: int = 100
    weight_decay: float = 0.01
    clip: float = 1.0
    eval_windows: int = 256
    n_evals: int = 4
    data_path: str = "/opt/data/enwik8"


def configure_arithmetic(regime: str) -> dict:
    """Make the Triton kernels the only place where the arithmetic differs between regimes."""
    import triton
    import triton.language as tl

    assert regime in REGIMES, regime
    # PyTorch's own TF32 switches off everywhere: the regimes differ only inside the kernels
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import fla.ops.gated_delta_rule.chunk_fwd as chunk_fwd
    if regime == "fp32_ieee":
        # this kernel selects TF32 explicitly for its triangular solve, which TRITON_F32_DEFAULT does not override
        chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    return {
        "triton_f32_default": str(triton.knobs.language.fp32_default),
        "solve_tril_precision": str(chunk_fwd.SOLVE_TRIL_DOT_PRECISION.value),
    }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class GatedDeltaBlock(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.norm1 = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.b_proj = nn.Linear(d, h, bias=False)
        self.a_proj = nn.Linear(d, h, bias=False)
        self.a_log = nn.Parameter(torch.log(torch.linspace(1.0, 4.0, h)))
        self.dt_bias = nn.Parameter(torch.linspace(-4.0, -1.0, h))
        self.g_proj = nn.Linear(d, d, bias=False)
        self.o_norm = RMSNorm(d // h)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, chunk: int, kernel_dtype: torch.dtype, recurrent: bool):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        bsz, t, d = x.shape
        y = self.norm1(x)
        heads = lambda z: z.view(bsz, t, self.h, d // self.h)
        q = F.normalize(heads(F.silu(self.q_proj(y))).float(), dim=-1)
        k = F.normalize(heads(F.silu(self.k_proj(y))).float(), dim=-1)
        v = heads(F.silu(self.v_proj(y)))
        beta = self.b_proj(y).float().sigmoid()
        g = -self.a_log.exp() * F.softplus(self.a_proj(y).float() + self.dt_bias)
        # every kernel input is cast to one format, so the kernel sees exactly the arithmetic the regime names
        q, k, v, g, beta = (z.to(kernel_dtype) for z in (q, k, v, g, beta))
        if recurrent:
            o = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)[0]
        else:
            o = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk)[0]
        o = self.o_norm(o).reshape(bsz, t, d).to(y.dtype) * F.silu(self.g_proj(y))
        x = x + self.o_proj(o)
        return x + self.mlp(self.norm2(x))


class ByteLM(nn.Module):
    def __init__(self, cfg: PilotConfig):
        super().__init__()
        self.embed = nn.Embedding(256, cfg.d_model)
        self.blocks = nn.ModuleList(GatedDeltaBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 256, bias=False)

    def forward(self, tokens, chunk: int, regime: str, reference: bool = False):
        """`reference=True` is the common evaluation pass: float32, recurrent kernel, no autocast."""
        low = regime == "bf16" and not reference
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=low):
            x = self.embed(tokens)
            for block in self.blocks:
                x = block(x, chunk, torch.bfloat16 if low else torch.float32, recurrent=reference)
            return self.head(self.norm(x)).float()


def load_data(path: str):
    raw = torch.from_numpy(np.fromfile(path, dtype=np.uint8)).cuda()
    assert raw.numel() == 100_000_000, raw.numel()
    return raw[:90_000_000], raw[90_000_000:95_000_000]


def windows(data, starts, seq_len):
    idx = starts[:, None] + torch.arange(seq_len + 1, device=data.device)
    return data[idx].long()


@torch.no_grad()
def evaluate(model, val, cfg: PilotConfig, reference: bool) -> float:
    model.eval()
    starts = torch.linspace(0, val.numel() - cfg.seq_len - 2, cfg.eval_windows).long().cuda()
    total = 0.0
    for i in range(0, cfg.eval_windows, cfg.batch):
        tok = windows(val, starts[i:i + cfg.batch], cfg.seq_len)
        logits = model(tok[:, :-1], cfg.chunk, cfg.regime, reference=reference)
        total += F.cross_entropy(logits.flatten(0, 1).double(), tok[:, 1:].flatten(), reduction="sum").item()
    model.train()
    return total / (cfg.eval_windows * cfg.seq_len)


def kernel_oracle_deviation(cfg: PilotConfig) -> float:
    """How far the kernel, in this run's arithmetic and chunk length, is from the float64 recurrence."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    from ..equivalence.core import deviation
    from ..equivalence.kernels import oracles
    from ..equivalence.kernels.cases import gated_inputs

    dtype = torch.bfloat16 if cfg.regime == "bf16" else torch.float32
    x = {n: t.to(dtype) for n, t in gated_inputs(torch.Generator().manual_seed(11), 1024).items()}
    with torch.no_grad():
        out = chunk_gated_delta_rule(x["q"].cuda(), x["k"].cuda(), x["v"].cuda(), x["g"].cuda(), x["beta"].cuda(),
                                     chunk_size=cfg.chunk)[0].cpu()
        ref = oracles.delta_oracle({n: t.double() for n, t in x.items()}, cfg.chunk)
    return deviation(ref, out)["max_rel"]


def run(cfg: PilotConfig, log=print) -> list[dict]:
    arithmetic = configure_arithmetic(cfg.regime)
    base = {**asdict(cfg), **arithmetic, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "kernel_oracle_dev": kernel_oracle_deviation(cfg)}
    log(f"{cfg.regime} chunk={cfg.chunk} seed={cfg.seed} rep={cfg.replicate}  {arithmetic}  "
        f"kernel-vs-oracle {base['kernel_oracle_dev']:.2e}")

    train, val = load_data(cfg.data_path)
    torch.manual_seed(cfg.seed)
    model = ByteLM(cfg).cuda()
    base["params"] = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    schedule = lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps)))
    sampler = torch.Generator().manual_seed(1000 + cfg.seed)  # the order of training windows
    eval_at = {round(cfg.steps * (i + 1) / cfg.n_evals) for i in range(cfg.n_evals)}

    rows, running, count, timed_tokens, timed_seconds = [], 0.0, 0, 0, 0.0
    start = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        tic = time.perf_counter()
        starts = torch.randint(0, train.numel() - cfg.seq_len - 1, (cfg.batch,), generator=sampler).cuda()
        tok = windows(train, starts, cfg.seq_len)
        for group in opt.param_groups:
            group["lr"] = cfg.lr * schedule(step - 1)
        logits = model(tok[:, :-1], cfg.chunk, cfg.regime)
        loss = F.cross_entropy(logits.flatten(0, 1), tok[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        running += loss.item()
        count += 1
        if step > 20:  # exclude kernel compilation from the throughput figure
            timed_tokens += cfg.batch * cfg.seq_len
            timed_seconds += time.perf_counter() - tic
        if not math.isfinite(loss.item()):
            rows.append({**base, "step": step, "status": "diverged"})
            return rows
        if step in eval_at:
            final = step == cfg.steps
            row = {**base, "step": step, "status": "ok", "final": final, "train_loss": running / count,
                   "val_loss_own": evaluate(model, val, cfg, reference=False),
                   "val_loss_ref": evaluate(model, val, cfg, reference=True),
                   "tokens_per_s": timed_tokens / timed_seconds if timed_seconds else float("nan"),
                   "wall_s": time.perf_counter() - start}
            if final:
                row["param_l2"] = math.sqrt(sum(p.double().pow(2).sum().item() for p in model.parameters()))
            rows.append(row)
            log(f"  step {step:5d}  train {row['train_loss']:.4f}  val own {row['val_loss_own']:.4f}  "
                f"val ref {row['val_loss_ref']:.4f}  {row['tokens_per_s'] / 1e3:.0f}k tok/s  {row['wall_s']:.0f}s")
            running, count = 0.0, 0
    return rows
