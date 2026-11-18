"""The float64 oracles used for the GPU kernel tier agree with the libraries' own reference ops.

Needs the pinned sources in third_party/ (fetched on first use); skipped if they cannot be fetched.
"""
import pytest
import torch

from src.equivalence.archs.common import l2norm
from src.equivalence.core import deviation
from src.equivalence.kernels import oracles
from src.equivalence.library import vendor

B, T, H, D = 2, 96, 2, 16
TOL = 1e-11


@pytest.fixture(scope="module")
def fla():
    try:
        vendor.ensure("fla")
        vendor.ensure("lact")
    except Exception as exc:  # no network, no git
        pytest.skip(f"pinned third-party sources unavailable: {exc}")
    return lambda rel: vendor.load("fla", rel, lift=True)


def rand(gen, *shape):
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


@pytest.fixture
def x():
    gen = torch.Generator().manual_seed(99)
    return {"q": l2norm(rand(gen, B, T, H, D)), "k": l2norm(rand(gen, B, T, H, D)), "v": rand(gen, B, T, H, D),
            "beta": torch.sigmoid(rand(gen, B, T, H)), "g": -0.1 * torch.nn.functional.softplus(rand(gen, B, T, H))}


def test_gated_delta_oracle_matches_library_recurrence(fla, x):
    with vendor.default_dtype(torch.float64):
        ref = fla("fla/ops/gated_delta_rule/naive.py").naive_recurrent_gated_delta_rule(
            x["q"], x["k"], x["v"], x["beta"], x["g"])[0]
    assert deviation(ref, oracles.delta_oracle(x, 0))["max_rel"] < TOL


def test_delta_oracle_matches_library_recurrence(fla, x):
    x = {n: t for n, t in x.items() if n != "g"}
    with vendor.default_dtype(torch.float64):  # this library op uses the [B, H, T, D] layout
        ref = fla("fla/ops/delta_rule/naive.py").delta_rule_recurrence(
            x["q"].transpose(1, 2), x["k"].transpose(1, 2), x["v"].transpose(1, 2), x["beta"].transpose(1, 2))[0]
    assert deviation(ref.transpose(1, 2), oracles.delta_oracle(x, 0))["max_rel"] < TOL


def test_simple_gla_oracle_matches_library_recurrence(fla, x):
    with vendor.default_dtype(torch.float64):
        ref = fla("fla/ops/simple_gla/naive.py").naive_recurrent_simple_gla(x["q"], x["k"], x["v"], x["g"])[0]
    assert deviation(ref, oracles.simple_gla_oracle(x, 0))["max_rel"] < TOL


def test_gla_oracle_matches_library_recurrence(fla, x):
    gen = torch.Generator().manual_seed(5)
    x = dict(x, g=torch.nn.functional.logsigmoid(rand(gen, B, T, H, D)) / 16)
    with vendor.default_dtype(torch.float64):
        ref = fla("fla/ops/gla/naive.py").naive_recurrent_gla(x["q"], x["k"], x["v"], x["g"])[0]
    assert deviation(ref, oracles.gla_oracle(x, 0))["max_rel"] < TOL


@pytest.mark.parametrize("chunk", [1, 16, 40, 96, 200])
def test_lact_oracle_matches_official_op(fla, chunk):
    gen = torch.Generator().manual_seed(3)
    n = B * H
    x = {"w0": rand(gen, n, D, D) / D ** 0.5, "w1": rand(gen, n, D, D) / D ** 0.5, "w2": rand(gen, n, D, D) / D ** 0.5,
         "q": l2norm(rand(gen, n, T, D)), "k": l2norm(rand(gen, n, T, D)), "v": rand(gen, n, T, D),
         "lr0": 0.01 * torch.rand(n, T, 1, generator=gen, dtype=torch.float64),
         "lr1": 0.01 * torch.rand(n, T, 1, generator=gen, dtype=torch.float64),
         "lr2": 0.01 * torch.rand(n, T, 1, generator=gen, dtype=torch.float64)}
    official = vendor.load("lact", "lact_llm/lact_model/ttt_operation.py")
    ref = official.block_causal_lact_swiglu(x["w0"], x["w1"], x["w2"], x["q"], x["k"], x["v"],
                                            x["lr0"], x["lr1"], x["lr2"], chunk_size=chunk)
    assert deviation(ref, oracles.lact_oracle(x, chunk))["max_rel"] < TOL
