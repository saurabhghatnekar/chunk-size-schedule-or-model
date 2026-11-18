"""Numerical checks of the propositions behind the scheduling class.

Each chunkwise form is compared with an independently written token-by-token
recurrence in float64. Chunk length 1 and chunk length = sequence are included,
together with a ragged tiling.
"""
import pytest
import torch

from src.equivalence.archs.delta import chunk_gated_delta, recurrent_gated_delta
from src.equivalence.archs.gla import chunk_gla, recurrent_gla
from src.equivalence.archs.mamba2 import chunk_ssd, recurrent_ssd
from src.equivalence.archs.softmax import tiled_causal_softmax_attention
from src.equivalence.archs.common import causal_softmax_attention, l2norm
from src.equivalence.core import deviation

T = 96
CHUNKS = [1, 2, 7, 16, 64, 96, 200]
TOL = 1e-11


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(1234)


def rand(gen, *shape):
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_gla_chunkwise_equals_recurrence(gen, chunk):
    q, k, v = rand(gen, 2, 2, T, 8), rand(gen, 2, 2, T, 8), rand(gen, 2, 2, T, 12)
    log_alpha = torch.nn.functional.logsigmoid(rand(gen, 2, 2, T, 8)) / 4
    ref = recurrent_gla(q, k, v, log_alpha)
    assert deviation(ref, chunk_gla(q, k, v, log_alpha, chunk))["max_rel"] < TOL


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("chunk", CHUNKS)
def test_delta_chunkwise_equals_recurrence(gen, chunk, gated):
    q, k, v = l2norm(rand(gen, 2, 2, T, 8)), l2norm(rand(gen, 2, 2, T, 8)), rand(gen, 2, 2, T, 12)
    beta = torch.sigmoid(rand(gen, 2, 2, T))
    g = -torch.nn.functional.softplus(rand(gen, 2, 2, T)) * 0.3 if gated else torch.zeros(2, 2, T, dtype=torch.float64)
    ref = recurrent_gated_delta(q, k, v, beta, g)
    assert deviation(ref, chunk_gated_delta(q, k, v, beta, g, chunk))["max_rel"] < TOL


@pytest.mark.parametrize("chunk", CHUNKS)
def test_ssd_chunkwise_equals_recurrence(gen, chunk):
    x, bmat, cmat = rand(gen, 2, T, 3, 5), rand(gen, 2, T, 3, 6), rand(gen, 2, T, 3, 6)
    a = -torch.nn.functional.softplus(rand(gen, 2, T, 3)) * 0.3
    ref = recurrent_ssd(x, a, bmat, cmat)
    assert deviation(ref, chunk_ssd(x, a, bmat, cmat, chunk))["max_rel"] < TOL


@pytest.mark.parametrize("tile", CHUNKS)
def test_tiled_softmax_equals_plain_softmax(gen, tile):
    q, k, v = rand(gen, 2, 2, T, 8), rand(gen, 2, 2, T, 8), rand(gen, 2, 2, T, 8)
    ref = causal_softmax_attention(q, k, v)
    assert deviation(ref, tiled_causal_softmax_attention(q, k, v, tile))["max_rel"] < TOL


def test_exact_family_gradients_agree(gen):
    """The backward pass re-associates too: gradients w.r.t. inputs agree across chunk lengths."""
    q = l2norm(rand(gen, 1, 2, T, 8)).requires_grad_()
    k = l2norm(rand(gen, 1, 2, T, 8)).requires_grad_()
    v = rand(gen, 1, 2, T, 12).requires_grad_()
    beta = torch.sigmoid(rand(gen, 1, 2, T)).requires_grad_()
    g = (-torch.nn.functional.softplus(rand(gen, 1, 2, T)) * 0.3).requires_grad_()
    probe = rand(gen, 1, 2, T, 12)
    leaves = [q, k, v, beta, g]
    ref = torch.autograd.grad((recurrent_gated_delta(*leaves) * probe).sum(), leaves)
    for chunk in [1, 16, 96]:
        got = torch.autograd.grad((chunk_gated_delta(*leaves, chunk) * probe).sum(), leaves)
        for r, t in zip(ref, got):
            assert deviation(r, t)["max_rel"] < TOL
