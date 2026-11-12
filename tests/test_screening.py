"""Checks on the deterministic screens."""

from pathlib import Path

import numpy as np
import pytest

from sso_sciml import screening
from sso_sciml.data import t1
from sso_sciml.physics.induction import IGData

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "t1_sample.npz"
P = {name: j for j, name in enumerate(t1.NAMES)}


def case(**changes) -> np.ndarray:
    base = {"k": 0.5, "w_r": 1.0, "rsc_hz": 10.0, "pll_hz": 10.0, "outer_ratio": 0.1, "ff": 0.5,
            "x_line": 1.0, "xr_line": 20.0, "x_t": 0.15, "par_ratio": 2.0, "n_online": 1.0,
            "r_r": 0.006}  # fmt: skip
    base.update(changes)
    return np.array([[base[n] for n in t1.NAMES]])


def test_radial_line_resonates_where_the_series_reactances_cancel():
    d = IGData()
    theta = case(par_ratio=1e6, xr_line=1e6)  # no parallel path to speak of, no resistance
    x_l = d.x_e + 0.15 + d.x_ls + d.x_lr
    expected = np.sqrt(0.5 * d.x_e / x_l)
    assert screening.network_resonance(theta)[0] == pytest.approx(expected, abs=2e-3)


def test_resonance_rises_with_compensation():
    w = [screening.network_resonance(case(k=k, par_ratio=1e6))[0] for k in (0.2, 0.5, 0.8)]
    assert w[0] < w[1] < w[2]


def test_a_strong_resistive_parallel_path_removes_the_resonance():
    theta = case(par_ratio=0.5, xr_line=10.0, k=0.3)
    assert screening.network_resonance(theta)[0] == 0.0
    # Without resistance the same network always shows a crossing.
    assert screening.network_resonance(theta, lossless=True)[0] > 0.0


def test_slip_score_is_zero_without_a_resonance_below_rotor_speed():
    assert screening.slip_score(case(par_ratio=0.5, xr_line=10.0, k=0.3))[0] == 0.0
    slow, fast = (screening.slip_score(case(par_ratio=1e6, w_r=w))[0] for w in (0.8, 1.2))
    assert slow > fast > 0.0  # the nearer the rotor speed to the resonance, the higher


def test_zero_miss_threshold_and_screen():
    score = np.array([-3.0, -1.0, 0.5, 2.0, -2.0])
    sigma = np.array([-1.0, 1.0, 1.0, 1.0, -1.0])
    tau = screening.zero_miss_threshold(score, sigma > 0)
    assert tau == -1.0
    out = screening.screen(score, sigma, tau)
    assert out == {"cleared": 0.4, "missed_unstable": 0.0, "false_alarm": 0.0}
    out = screening.screen(score, sigma, 0.0)
    assert out["missed_unstable"] == pytest.approx(1 / 3)


def test_phasor_model_does_not_see_an_unstable_sub_synchronous_mode():
    z = np.load(SAMPLE)
    unstable = np.where((z["solver"][:, 4] > 0) & (z["solver"][:, 0] > 5.0))[0][:3]
    assert len(unstable) == 3
    for row in unstable:
        a = t1.state_matrix(z["theta"][row])
        assert np.linalg.eigvals(a).real.max() > 5.0
        assert screening.phasor_growth(a) < 1.0
