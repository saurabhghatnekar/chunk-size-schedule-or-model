"""A small language model wrapped around one mixer, so "forward and backward"
means logits and the gradient of a next-token loss with respect to every weight.

The perturbed controls share their checkpoint with the clean Gated DeltaNet so
that the only difference between them is the injected defect.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .archs import SPECS, ArchSpec
from .archs.common import RMSNorm

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 256
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    seq_len: int = 512
    batch: int = 2
    seed: int = 0


class Block(nn.Module):
    def __init__(self, mixer: nn.Module, d_model: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = mixer
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, chunk_size):
        x = x + self.mixer(self.norm1(x), chunk_size)
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, spec: ArchSpec, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(spec.build(cfg.d_model, cfg.n_heads), cfg.d_model) for _ in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(self, tokens, chunk_size: int):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, chunk_size)
        return self.head(self.norm(x))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_name(spec: ArchSpec) -> str:
    return "gated_deltanet" if spec.name.startswith("perturbed_") else spec.name


def load_checkpoint(spec: ArchSpec, cfg: ModelConfig, create: bool = True):
    """Return (float64 state_dict, sha256). The file is created once and then reused."""
    path = CKPT_DIR / f"{checkpoint_name(spec)}.pt"
    if not path.exists():
        if not create:
            raise FileNotFoundError(path)
        CKPT_DIR.mkdir(exist_ok=True)
        torch.manual_seed(cfg.seed)
        model = TinyLM(SPECS[checkpoint_name(spec)], cfg).double()
        torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, path)
    blob = torch.load(path, weights_only=True)
    if blob["config"] != asdict(cfg):
        raise ValueError(f"{path} was created with {blob['config']}, not {asdict(cfg)}")
    return blob["state_dict"], _sha256(path)


def load_batch(cfg: ModelConfig):
    """Return (tokens, sha256). One fixed batch shared by every architecture."""
    path = CKPT_DIR / "batch.pt"
    if not path.exists():
        CKPT_DIR.mkdir(exist_ok=True)
        gen = torch.Generator().manual_seed(cfg.seed + 1)
        tokens = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq_len + 1), generator=gen)
        torch.save({"config": asdict(cfg), "tokens": tokens}, path)
    blob = torch.load(path, weights_only=True)
    if blob["config"] != asdict(cfg):
        raise ValueError(f"{path} was created with {blob['config']}, not {asdict(cfg)}")
    return blob["tokens"], _sha256(path)


def build_model(spec: ArchSpec, cfg: ModelConfig, state_dict, dtype: torch.dtype) -> TinyLM:
    model = TinyLM(spec, cfg).double()
    model.load_state_dict(state_dict)
    return model.to(dtype)


def forward_backward(model: TinyLM, tokens, chunk_size: int):
    """Logits, loss and the gradient of the loss w.r.t. every parameter."""
    model.zero_grad(set_to_none=True)
    logits = model(tokens[:, :-1], chunk_size)
    # the loss itself is always reduced in float64 so that the reduction is not
    # what the precision ladder measures
    loss = F.cross_entropy(logits.double().flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    # a parameter the loss does not depend on at this chunk length has gradient zero
    grads = {
        n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in model.named_parameters()
    }
    return logits.detach(), loss.item(), grads
