"""Unit tests of the harness itself: metrics, the decision rule, determinism, parameter counts."""
import torch

from src.equivalence.archs import SPECS
from src.equivalence.core import deviation, grad_deviation
from src.equivalence.model import ModelConfig, TinyLM, forward_backward
from src.equivalence.precision_scaling import classify, loglog_slope, verdict


def test_deviation_metrics():
    ref = torch.tensor([1.0, -4.0, 0.0])
    out = deviation(ref, ref + torch.tensor([0.0, 0.0, 0.5]))
    assert out["max_abs"] == 0.5 and out["max_rel"] == 0.125
    assert deviation(ref, ref)["max_rel"] == 0.0


def test_grad_deviation_reports_the_worst_tensor():
    ref = {"a": torch.ones(3), "b": torch.full((2,), 1e-3)}
    test = {"a": torch.ones(3), "b": torch.full((2,), 2e-3)}
    out = grad_deviation(ref, test)
    assert out["worst_tensor"] == "b" and abs(out["max_rel_per_tensor"] - 1.0) < 1e-9
    assert abs(out["max_rel"] - 1e-3) < 1e-9


def test_decision_rule():
    assert verdict(0.0, 0.0) == "ROUNDING"
    assert verdict(3e-6, 4e-15) == "ROUNDING"          # collapses with precision
    assert verdict(2e-1, 2e-1) == "STRUCTURAL"         # precision-invariant
    assert verdict(3e-6, 3e-6) == "STRUCTURAL"         # small but precision-invariant: still structural
    assert verdict(4e-6, 5e-9) == "INCONCLUSIVE"       # too big for rounding, shrinks too much for structure
    assert verdict(1e-5, 1e-8) == "INCONCLUSIVE"
    assert verdict(1e-6, 1e-8) == "STRUCTURAL"          # shrink of exactly PERSIST_MAX still counts as persisting
    assert classify("ROUNDING", "exact") == "scheduling"
    assert classify("STRUCTURAL", "boundary") == "architectural"
    assert classify("STRUCTURAL", "exact") == "implementation bug"
    assert classify("INCONCLUSIVE", "exact") == "unresolved"


def test_loglog_slope():
    eps = [2.0 ** -8, 2.0 ** -11, 2.0 ** -24, 2.0 ** -53]
    assert abs(loglog_slope(eps, [50 * e for e in eps]) - 1.0) < 1e-9
    assert abs(loglog_slope(eps, [0.3] * 4)) < 1e-9


def test_same_configuration_twice_is_bitwise_identical():
    cfg = ModelConfig(seq_len=64)
    tokens = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq_len + 1), generator=torch.Generator().manual_seed(0))
    for name in ["gated_deltanet", "ttt_titans", "lact", "flash", "hla", "mamba2", "gla", "softmax_tile"]:
        torch.manual_seed(0)
        model = TinyLM(SPECS[name], cfg).double()
        a = forward_backward(model, tokens, 16)
        b = forward_backward(model, tokens, 16)
        assert torch.equal(a[0], b[0]), name
        assert all(torch.equal(a[2][k], b[2][k]) for k in a[2]), name


def test_parameter_count_does_not_depend_on_chunk_length():
    cfg = ModelConfig(seq_len=64)
    tokens = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq_len + 1), generator=torch.Generator().manual_seed(0))
    for name, spec in SPECS.items():
        model = TinyLM(spec, cfg).double()
        before = sum(p.numel() for p in model.parameters())
        for chunk in (16, 64):
            model(tokens[:, :-1], chunk)
        assert sum(p.numel() for p in model.parameters()) == before, name
