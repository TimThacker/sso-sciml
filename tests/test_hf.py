"""Control delay and a filter capacitor in the converter; modes at infinity."""

import numpy as np
import pytest

from sso_sciml import admittance as adm
from sso_sciml import hf
from sso_sciml import vectorfit as vf
from sso_sciml.data import c1

COL = {name: j for j, name in enumerate(hf.COLUMNS)}
THETA = c1.sample(32, "pll")


def test_closing_on_the_network_returns_no_mode_at_infinity():
    # The algebraic equations of the network give eigenvalues at infinity. They used to
    # come back now and then as finite numbers around 1e18 and were read as modes.
    worst = 0.0
    for theta in THETA:
        model, x0, p, net = c1.operating_point(theta, 0.0)
        plant, _ = hf.solve(theta, 0.0, 0.0, 0.0, 0.0)
        modes = vf.closed_loop_modes(plant.a, plant.b, plant.c, plant.d, net)
        assert np.abs(modes).max() < vf.MODE_LIMIT
        assert modes.size == 14  # as many as the full model has states
        worst = max(worst, abs(modes.real.max() - c1.growth(model, x0, p)[0]))
    assert worst < 1e-7


def test_modes_with_a_delay_are_zeros_of_the_loop_determinant():
    # The delay states carry 12 / T^2. Before the plant was balanced the generalised
    # eigenvalue problem lost up to 0.4 1/s on the least-damped mode (step 30).
    for theta in THETA[:8]:
        for t_d in (5e-5, 2e-4, 1e-3):
            plant, net = hf.solve(theta, 0.0, t_d, 0.0, 0.0)
            lam = vf.closed_loop_modes(plant.a, plant.b, plant.c, plant.d, net)
            top = lam[np.argmax(lam.real)]

            def det(s, plant=plant, net=net):
                s = np.atleast_1d(s)
                y, z = plant.admittance(s), adm.network_impedance(net, s)
                return np.linalg.det(np.eye(2) + y @ z)[0]

            h = 1e-6 * (1 + abs(top))
            step = det(top) / ((det(top + h) - det(top - h)) / (2 * h))
            assert abs(step) < 1e-6  # one Newton step from the mode goes nowhere


def test_delay_approximant_has_unit_gain_and_the_phase_of_a_delay():
    t = 5e-4
    a = np.array([[0.0, 1.0], [-12.0 / t**2, -6.0 / t]])
    b, c = np.array([0.0, 1.0]), np.array([0.0, -12.0 / t])
    for f in (10.0, 100.0, 400.0):
        s = 2j * np.pi * f
        h = 1.0 + c @ np.linalg.solve(s * np.eye(2) - a, b)
        assert abs(h) == pytest.approx(1.0, abs=1e-9)
        assert np.angle(h) == pytest.approx(-2 * np.pi * f * t, abs=0.02)


def test_delayed_converter_tends_to_the_plain_one():
    theta = THETA[0]
    plain, _ = hf.solve(theta, 0.0, 0.0, 0.0, 0.0)
    s = 2j * np.pi * np.array([1.0, 20.0, 100.0])
    gap = []
    for t_d in (1e-5, 1e-6):
        delayed, _ = hf.solve(theta, 0.0, t_d, 0.0, 0.0)
        gap.append(np.abs(delayed.admittance(s) - plain.admittance(s)).max())
    # The difference shrinks in proportion to the delay.
    assert gap[1] < 5e-3
    assert gap[1] == pytest.approx(gap[0] / 10, rel=0.05)


def test_capacitor_branch_adds_its_own_admittance():
    theta, b_c, r_d = THETA[1], 0.05, 0.1
    with_cap, _ = hf.solve(theta, 0.0, 0.0, b_c, r_d)
    s = 2j * np.pi * np.array([5.0, 50.0, 500.0])
    sig = s / hf.WB
    # A capacitor in the synchronous frame: reactance 1 / (b (sigma +- j)) per sequence.
    plus, minus = 1 / (r_d + 1 / (b_c * (sig + 1j))), 1 / (r_d + 1 / (b_c * (sig - 1j)))
    y_d, y_q = 0.5 * (plus + minus), (plus - minus) / 2j
    branch = np.stack([np.stack([y_d, -y_q], -1), np.stack([y_q, y_d], -1)], -2)
    # The converter part is whatever is left; it must be the converter at that operating
    # point, which has no pole of the branch. Check the branch alone through its matrices.
    n = with_cap.a.shape[0] - 2
    own = np.array([
        with_cap.c[:, n:] @ np.linalg.solve(v * np.eye(2) - with_cap.a[n:, n:], with_cap.b[n:])
        for v in s
    ]) + with_cap.d  # fmt: skip
    assert np.allclose(own, branch, atol=1e-9)


def test_case_row_and_the_base_variant():
    row = hf.case(("pll", THETA[2]))
    assert row.shape == (len(hf.COLUMNS),)
    model, x0, p, _ = c1.operating_point(THETA[2], 0.0)
    assert row[COL["top_base"]] == pytest.approx(c1.growth(model, x0, p)[0], abs=1e-7)
    assert row[COL["alone_both"]] < 0


def test_filter_resonance_formula():
    theta = THETA[3]
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    x_grid = abs(hf._z60(c1.network(th)).imag)
    f = hf.lcl_frequency(theta, 0.05)
    par = th["x_f"] * x_grid / (th["x_f"] + x_grid)
    assert f == pytest.approx(60.0 / np.sqrt(par * 0.05))
    assert adm.scan_grid(f_max=10000.0).max() == 10000.0
