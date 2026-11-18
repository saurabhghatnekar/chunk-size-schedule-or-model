"""The pre-registered decision rule of the precision pilot (docs/go_no_go.md, section 1)."""
from src.pilot.analyze import decide


def test_consistent_shift_larger_than_seed_spread_is_an_effect():
    verdict, mean, sd, t = decide([0.010, 0.011, 0.009], sigma_seed=0.002)
    assert verdict == "EFFECT" and abs(mean - 0.010) < 1e-12 and t > 4.30


def test_scatter_around_zero_is_not_an_effect():
    assert decide([0.010, -0.010, 0.002], sigma_seed=0.002)[0] == "NO EFFECT DETECTED"


def test_consistent_but_smaller_than_seed_spread_is_not_an_effect():
    assert decide([0.0010, 0.0011, 0.0009], sigma_seed=0.02)[0] == "NO EFFECT DETECTED"
