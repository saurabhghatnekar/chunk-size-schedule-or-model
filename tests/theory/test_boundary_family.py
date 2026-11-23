"""Numerical checks of the propositions behind the architectural class.

For each mechanism: the limiting case in which the layer reduces to something
chunk-free, and a check that away from that limit the chunk length changes the
function by an amount far above rounding.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from src.equivalence.archs.common import l2norm
from src.equivalence.archs.delta import DeltaMixer, recurrent_gated_delta
from src.equivalence.archs.flash import FlashMixer
from src.equivalence.archs.hla import HLAMixer
from src.equivalence.archs.lact import LaCTMixer, block_causal_lact_swiglu
from src.equivalence.archs.ttt import minibatch_ttt_linear
from src.equivalence.core import deviation

T = 96
TOL = 1e-11
STRUCTURAL = 1e-4


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(4321)


def rand(gen, *shape):
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


# ---- test-time training, Titans-style -------------------------------------------------

def _ttt_inputs(gen):
    q, k, v = l2norm(rand(gen, 2, 2, T, 8)), l2norm(rand(gen, 2, 2, T, 8)), rand(gen, 2, 2, T, 12)
    theta = 0.5 * torch.sigmoid(rand(gen, 2, 2, T))
    return q, k, v, theta


def test_ttt_chunk_one_without_momentum_is_the_delta_rule(gen):
    q, k, v, theta = _ttt_inputs(gen)
    zeros = torch.zeros_like(theta)
    got = minibatch_ttt_linear(q, k, v, theta, zeros, zeros, chunk_size=1)
    ref = recurrent_gated_delta(q, k, v, beta=2 * theta, g=zeros)
    assert deviation(ref, got)["max_rel"] < TOL


def test_ttt_single_chunk_without_momentum_is_hebbian_linear_attention(gen):
    """With every gradient evaluated at M_0 = 0 the update is M_t = M_{t-1} + 2 theta_t k_t^T v_t."""
    q, k, v, theta = _ttt_inputs(gen)
    zeros = torch.zeros_like(theta)
    got = minibatch_ttt_linear(q, k, v, theta, zeros, zeros, chunk_size=T)
    attn = torch.tril(q @ k.transpose(-1, -2)) * (2 * theta)[..., None, :]
    assert deviation(attn @ v, got)["max_rel"] < TOL


def test_ttt_chunk_length_changes_the_function(gen):
    q, k, v, theta = _ttt_inputs(gen)
    eta, alpha = torch.sigmoid(rand(gen, 2, 2, T) - 1), torch.sigmoid(rand(gen, 2, 2, T) - 4)
    outs = [minibatch_ttt_linear(q, k, v, theta, eta, alpha, c) for c in (1, 8, 32)]
    assert deviation(outs[0], outs[1])["max_rel"] > STRUCTURAL
    assert deviation(outs[1], outs[2])["max_rel"] > STRUCTURAL
    # tokens before the first boundary that differs see identical computation
    assert deviation(outs[1][:, :, :8], outs[2][:, :, :8])["max_rel"] < TOL


# ---- LaCT ------------------------------------------------------------------------------

def test_lact_manual_gradient_matches_autograd(gen):
    """One apply-then-update step: the hand-written fast-weight gradient is the autograd gradient
    of sum_t lr_t <f(k_t), v_t> (with one learning rate shared by the three fast weights)."""
    b, d, chunk = 2, 6, 5
    w0, w1, w2 = (rand(gen, b, d, d).requires_grad_() for _ in range(3))
    k, v = rand(gen, b, 2 * chunk, d), rand(gen, b, 2 * chunk, d)
    q = rand(gen, b, 2 * chunk, d)
    lr = F.softplus(rand(gen, b, 2 * chunk, 1))
    kc, vc = k[:, :chunk].transpose(1, 2), v[:, :chunk].transpose(1, 2)
    inner = ((w1 @ (F.silu(w0 @ kc) * (w2 @ kc))) * vc * lr[:, :chunk].transpose(1, 2)).sum()
    g0, g1, g2 = torch.autograd.grad(inner, [w0, w1, w2])
    norm = lambda w: w.norm(dim=2, keepdim=True)
    new = [(w + gw) / (norm(w + gw) + 1e-5) * norm(w) for w, gw in ((w0, g0), (w1, g1), (w2, g2))]
    qc = q[:, chunk:].transpose(1, 2)
    expected = (new[1] @ (F.silu(new[0] @ qc) * (new[2] @ qc))).transpose(1, 2)
    got = block_causal_lact_swiglu(w0, w1, w2, q, k, v, lr, lr, lr, chunk)[:, chunk:]
    assert deviation(expected, got)["max_rel"] < TOL


def test_lact_chunk_at_least_sequence_never_updates(gen):
    torch.manual_seed(0)
    mixer = LaCTMixer(32, 2).double()
    x = rand(gen, 2, T, 32)
    assert deviation(mixer(x, T), mixer(x, 4 * T))["max_rel"] == 0.0
    assert deviation(mixer(x, T), mixer(x, 16))["max_rel"] > STRUCTURAL


# ---- FLASH -----------------------------------------------------------------------------

def test_flash_single_chunk_is_pure_quadratic_attention(gen):
    torch.manual_seed(0)
    mixer = FlashMixer(32, 2).double()
    x = rand(gen, 2, T, 32)
    u, v, z = F.silu(mixer.to_u(x)), F.silu(mixer.to_v(x)), F.silu(mixer.to_z(x))
    qq, qk = z * mixer.gamma[0] + mixer.beta[0], z * mixer.gamma[1] + mixer.beta[1]
    idx = torch.arange(T)
    off = idx[:, None] - idx[None, :]
    kernel = F.relu(qq @ qk.transpose(-1, -2) / T + mixer.rel_bias[off.clamp(min=0)]).pow(2)
    ref = mixer.o_proj(u * (kernel.masked_fill(off < 0, 0.0) @ v))
    assert deviation(ref, mixer(x, T))["max_rel"] < TOL
    assert deviation(mixer(x, T), mixer(x, 16))["max_rel"] > STRUCTURAL


# ---- HLA -------------------------------------------------------------------------------

def _hla_and_gdn(force_unit_routing):
    torch.manual_seed(0)
    hla = HLAMixer(32, 2, pool_window=8, force_unit_routing=force_unit_routing).double()
    gdn = DeltaMixer(32, 2, gated=True).double()
    gdn.load_state_dict({k: v for k, v in hla.state_dict().items() if k in gdn.state_dict()})
    return hla, gdn


@pytest.mark.parametrize("chunk", [8, 16, 40, 96])
def test_hla_with_unit_routing_is_gated_deltanet_at_every_chunk_length(gen, chunk):
    hla, gdn = _hla_and_gdn(force_unit_routing=True)
    x = rand(gen, 2, T, 32)
    assert deviation(gdn(x, 16), hla(x, chunk))["max_rel"] < TOL


def test_hla_with_the_released_gate_bias_still_depends_on_the_chunk_boundary(gen):
    # the authors' code adds 2.2 inside the routing sigmoid; gates move toward one but do not reach it
    hla = HLAMixer(32, 2, pool_window=8, gate_bias=2.2).double()
    x = torch.randn(2, 64, 32, generator=gen, dtype=torch.float64)
    assert deviation(hla(x, 16), hla(x, 32))["max_rel"] > 1e-3


def test_hla_single_chunk_is_gated_deltanet_and_routing_makes_the_boundary_matter(gen):
    hla, gdn = _hla_and_gdn(force_unit_routing=False)
    x = rand(gen, 2, T, 32)
    assert deviation(gdn(x, 16), hla(x, T))["max_rel"] < TOL
    assert deviation(hla(x, 16), hla(x, 32))["max_rel"] > STRUCTURAL
