"""Small pilot on the architectural class (docs/go_no_go_architectural.md).

A byte-level language model whose blocks are LaCT-style layers: a causal
sliding-window softmax attention plus the official LaCT fast-weight op, summed.
The attention window is fixed, so the test-time-training chunk length is the only
thing that differs between cells. A chunk length of at least the sequence length
means the official op never updates its fast weights: the same model with
test-time training switched off.

Needs a CUDA GPU and the pinned LaCT source (see src/equivalence/library/vendor.py).
"""
from __future__ import annotations

import functools
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .precision import RMSNorm, load_data, windows


@dataclass(frozen=True)
class ArchConfig:
    chunk: int
    seed: int
    replicate: int = 0  # same seed run again, to measure the run-to-run floor
    steps: int = 2000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    seq_len: int = 1024
    window: int = 64
    batch: int = 32
    lr: float = 1e-3
    warmup: int = 100
    weight_decay: float = 0.01
    clip: float = 1.0
    base_fast_lr: float = 0.001  # the LaCT repository's default
    eval_windows: int = 256
    n_evals: int = 4
    eval_chunks: tuple = (32, 128, 512, 1024)
    data_path: str = "/opt/data/enwik8"


@functools.lru_cache(maxsize=1)
def lact_op():
    """The official op, with its bfloat16 autocast decorator kept (torch.compile is disabled by the environment)."""
    from ..equivalence.library import vendor

    return vendor.load("lact", "lact_llm/lact_model/ttt_operation.py", strip=False).block_causal_lact_swiglu


class Rotary(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        half = dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(half) / half))
        angles = torch.arange(max_len)[:, None] * freqs[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):  # (B,H,T,D)
        t, half = x.shape[-2], x.shape[-1] // 2
        cos, sin = self.cos[:t].to(x.dtype), self.sin[:t].to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class LaCTBlock(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        hd = d // h
        self.h = h
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.rotary = Rotary(hd, cfg.seq_len)
        self.w0 = nn.Parameter(torch.randn(h, hd, hd) / math.sqrt(hd))
        self.w1 = nn.Parameter(torch.randn(h, hd, hd) / math.sqrt(hd))
        self.w2 = nn.Parameter(torch.randn(h, hd, hd) / math.sqrt(hd))
        self.lr_proj = nn.Linear(d, 3 * h)
        self.base_lr_inv = math.log(math.expm1(cfg.base_fast_lr))
        self.qk_scale = nn.Parameter(torch.ones(d, 2))
        self.qk_offset = nn.Parameter(torch.zeros(d, 2))
        self.ttt_norm = RMSNorm(hd)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, chunk: int, band):
        bsz, t, d = x.shape
        h = self.h
        y = self.norm1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        heads = lambda z: z.view(bsz, t, h, d // h).transpose(1, 2)  # (B,H,T,hd)

        attn = F.scaled_dot_product_attention(self.rotary(heads(q)), self.rotary(heads(k)), heads(v), attn_mask=band)
        attn = attn.transpose(1, 2).reshape(bsz, t, d)

        fast = lambda z: heads(z).reshape(bsz * h, t, d // h)  # heads folded into the batch, as in the LaCT layer
        l2 = lambda z: (z / (z.float().norm(dim=-1, keepdim=True) + 1e-5)).to(torch.bfloat16)
        fq = l2(F.silu(fast(q * self.qk_scale[:, 0] + self.qk_offset[:, 0])))
        fk = l2(F.silu(fast(k * self.qk_scale[:, 1] + self.qk_offset[:, 1])))
        fv = F.silu(fast(v)).to(torch.bfloat16)
        lr = F.softplus(self.lr_proj(y).float() + self.base_lr_inv)  # (B,T,3H), float32
        lr0, lr1, lr2 = lr.view(bsz, t, h, 3).transpose(1, 2).reshape(bsz * h, t, 3).split(1, dim=-1)
        # fast weights in float32, q/k/v in bfloat16, learning rates in float32: what the op's docstring prescribes
        fw = lact_op()(self.w0.float().repeat(bsz, 1, 1), self.w1.float().repeat(bsz, 1, 1),
                       self.w2.float().repeat(bsz, 1, 1), fq, fk, fv, lr0, lr1, lr2, chunk_size=chunk)
        fw = self.ttt_norm(fw).view(bsz, h, t, d // h).transpose(1, 2).reshape(bsz, t, d)

        x = x + self.o_proj(attn + fw.to(attn.dtype))
        return x + self.mlp(self.norm2(x))


class LaCTByteLM(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.embed = nn.Embedding(256, cfg.d_model)
        self.blocks = nn.ModuleList(LaCTBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 256, bias=False)
        idx = torch.arange(cfg.seq_len)
        gap = idx[:, None] - idx[None, :]
        self.register_buffer("band", (gap >= 0) & (gap < cfg.window), persistent=False)

    def forward(self, tokens, chunk: int):
        t = tokens.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.embed(tokens)
            for block in self.blocks:
                x = block(x, chunk, self.band[:t, :t])
            return self.head(self.norm(x)).float()


@torch.no_grad()
def evaluate(model, val, cfg: ArchConfig, chunk: int) -> float:
    model.eval()
    starts = torch.linspace(0, val.numel() - cfg.seq_len - 2, cfg.eval_windows).long().cuda()
    total = 0.0
    for i in range(0, cfg.eval_windows, cfg.batch):
        tok = windows(val, starts[i:i + cfg.batch], cfg.seq_len)
        logits = model(tok[:, :-1], chunk)
        total += F.cross_entropy(logits.flatten(0, 1).double(), tok[:, 1:].flatten(), reduction="sum").item()
    model.train()
    return total / (cfg.eval_windows * cfg.seq_len)


def run(cfg: ArchConfig, log=print) -> list[dict]:
    base = {**{k: v for k, v in asdict(cfg).items() if k != "eval_chunks"},
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    train, val = load_data(cfg.data_path)
    torch.manual_seed(cfg.seed)
    model = LaCTByteLM(cfg).cuda()
    base["params"] = sum(p.numel() for p in model.parameters())
    updates = len(range(0, cfg.seq_len - cfg.chunk, cfg.chunk))
    base["fast_weight_updates_per_sequence"] = updates
    log(f"lact chunk={cfg.chunk} seed={cfg.seed}  params {base['params']:,}  fast-weight updates per sequence {updates}")

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    schedule = lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps)))
    sampler = torch.Generator().manual_seed(1000 + cfg.seed)
    eval_at = {round(cfg.steps * (i + 1) / cfg.n_evals) for i in range(cfg.n_evals)}

    rows, running, count, timed_tokens, timed_seconds = [], 0.0, 0, 0, 0.0
    start = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        tic = time.perf_counter()
        starts = torch.randint(0, train.numel() - cfg.seq_len - 1, (cfg.batch,), generator=sampler).cuda()
        tok = windows(train, starts, cfg.seq_len)
        for group in opt.param_groups:
            group["lr"] = cfg.lr * schedule(step - 1)
        loss = F.cross_entropy(model(tok[:, :-1], cfg.chunk).flatten(0, 1), tok[:, 1:].flatten())
        if not math.isfinite(loss.item()):
            rows.append({**base, "step": step, "status": "diverged", "final": False})
            log(f"  diverged at step {step}")
            return rows
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        running += loss.item()
        count += 1
        if step > 20:
            timed_tokens += cfg.batch * cfg.seq_len
            timed_seconds += time.perf_counter() - tic
        if step in eval_at:
            final = step == cfg.steps
            row = {**base, "step": step, "status": "ok", "final": final, "train_loss": running / count,
                   "val_loss": evaluate(model, val, cfg, cfg.chunk),
                   "tokens_per_s": timed_tokens / timed_seconds if timed_seconds else float("nan"),
                   "wall_s": time.perf_counter() - start}
            if final:  # the same trained model evaluated at every chunk length
                for chunk in cfg.eval_chunks:
                    row[f"val_loss_at_chunk_{chunk}"] = (row["val_loss"] if chunk == cfg.chunk
                                                         else evaluate(model, val, cfg, chunk))
            rows.append(row)
            log(f"  step {step:5d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
                f"{row['tokens_per_s'] / 1e3:.0f}k tok/s  {row['wall_s']:.0f}s")
            running, count = 0.0, 0
    return rows


def benchmark(cfg: ArchConfig, steps: int = 30, warmup: int = 5) -> dict:
    """Training throughput of one configuration on random tokens: forward, backward and optimiser step.
    No data and no learning; it only times the computation the real run would do."""
    torch.manual_seed(0)
    out = {**{k: v for k, v in asdict(cfg).items() if k != "eval_chunks"}, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__}
    try:
        model = LaCTByteLM(cfg).cuda()
        out["params"] = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        tokens, seconds = 0, 0.0
        for step in range(steps):
            tok = torch.randint(0, 256, (cfg.batch, cfg.seq_len + 1), device="cuda")
            torch.cuda.synchronize()
            tic = time.perf_counter()
            loss = F.cross_entropy(model(tok[:, :-1], cfg.chunk).flatten(0, 1), tok[:, 1:].flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if step >= warmup:
                tokens += cfg.batch * cfg.seq_len
                seconds += time.perf_counter() - tic
        out.update(status="ok", tokens_per_s=tokens / seconds, peak_memory_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    except torch.OutOfMemoryError:
        out.update(status="out of memory")
    return out
