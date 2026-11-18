"""The pre-registered decision rule of the precision pilot (docs/go_no_go.md, section 1)."""
from src.pilot.analyze import decide


def test_consistent_shift_larger_than_seed_spread_is_an_effect():
    verdict, mean, sd, t = decide([0.010, 0.011, 0.009], sigma_seed=0.002)
    assert verdict == "EFFECT" and abs(mean - 0.010) < 1e-12 and t > 4.30


def test_scatter_around_zero_is_not_an_effect():
    assert decide([0.010, -0.010, 0.002], sigma_seed=0.002)[0] == "NO EFFECT DETECTED"


def test_consistent_but_smaller_than_seed_spread_is_not_an_effect():
    assert decide([0.0010, 0.0011, 0.0009], sigma_seed=0.02)[0] == "NO EFFECT DETECTED"


# ---- architectural pilot (docs/go_no_go_architectural.md) ----

from src.pilot.analyze_architectural import anova, decide as decide_arch  # noqa: E402


def test_anova_matches_a_hand_computed_case():
    f, s_within, df = anova({1: [1.0, 2.0, 3.0], 2: [2.0, 3.0, 4.0], 3: [5.0, 6.0, 7.0]})
    # grand mean 11/3; between = 3*((2-11/3)^2+(3-11/3)^2+(6-11/3)^2) = 26; within = 6; F = (26/2)/(6/6) = 13
    assert abs(f - 13.0) < 1e-9 and abs(s_within - 1.0) < 1e-9 and df == 6


def _cells(shift):
    base = [1.200, 1.202, 1.198, 1.201, 1.199]
    return {32: base, 128: [v + shift for v in base], 512: [v + 2 * shift for v in base]}


def test_clear_separation_is_an_effect():
    assert decide_arch(_cells(0.02), anchor=[1.30, 1.31, 1.29, 1.30, 1.30])["verdict"] == "EFFECT"


def test_overlapping_cells_are_not_an_effect():
    assert decide_arch(_cells(0.0002), anchor=[1.30, 1.31, 1.29, 1.30, 1.30])["verdict"] == "NO EFFECT DETECTED"


def test_no_gap_to_the_anchor_is_uninformative():
    assert decide_arch(_cells(0.0002), anchor=[1.200, 1.201, 1.199, 1.202, 1.198])["verdict"] == "UNINFORMATIVE"
