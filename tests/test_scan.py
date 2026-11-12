"""The doubly-fed plant as an admittance scan, and the tests on the loop eigenvalues."""

import numpy as np
import pytest

from sso_sciml import admittance as adm
from sso_sciml.data import t1
from sso_sciml.solver.linearize import jacobian

THETA = t1.sample(12)


@pytest.fixture(scope="module")
def cases():
    out = []
    for theta in THETA:
        model, x0, p = t1.operating_point(theta)
        lam = np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy())
        out.append((theta, lam, *t1.scan_pair(theta)))
    return out


def test_plant_cut_from_its_network_is_at_rest():
    model, x0, p = t1.operating_point(THETA[0])
    assert adm.DFIGScan(model, x0, p).residual < 1e-9


def test_scan_determinant_vanishes_at_the_modes_of_the_full_model():
    theta = THETA[0]
    model, x0, p = t1.operating_point(theta)
    lam = np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy())
    lam = lam[np.abs(lam.imag) > 1.0]
    plant = adm.DFIGScan(model, x0, p)
    loop = plant.admittance(lam) @ adm.network_impedance(t1.network(theta), lam)
    assert np.abs(np.linalg.det(np.eye(2) + loop)).max() < 1e-8


def test_nyquist_count_equals_the_number_of_unstable_modes(cases):
    for _, lam, y, z, poles in cases:
        assert adm.unstable_modes(y, z, poles) == int((lam.real > 0).sum())


def test_loop_scores_are_ordered(cases):
    for _, _, y, z, _ in cases:
        s = adm.loop_scores(y, z, adm.GUARDS)
        assert s["radius"] <= s["norm"] + 1e-12
        guards = [s[f"guard_{g:g}"] for g in adm.GUARDS]
        assert all(a <= b + 1e-12 for a, b in zip(guards[:-1], guards[1:], strict=True))


def test_gain_margin_below_one_only_on_stable_cases(cases):
    cleared = 0
    for _, lam, y, z, poles in cases:
        if poles == 0 and adm.loop_scores(y, z, (10.0,))["guard_10"] < 1:
            cleared += 1
            assert lam.real.max() < 0
    assert cleared > 0


def test_passive_plant_scores_zero():
    s = 2j * np.pi * adm.scan_grid(n=50)
    y = np.broadcast_to(np.eye(2) * 0.5, (s.size, 2, 2))
    z = np.broadcast_to(np.eye(2) * 3.0, (s.size, 2, 2))
    scores = adm.loop_scores(y, z, (30.0,))
    assert scores["norm"] == 0.0 and scores["radius"] == 0.0 and scores["guard_30"] == 0.0


def test_screen_scores_match_their_columns_and_repeat(cases):
    _, _, y, z, poles = cases[0]
    a = adm.screen_scores(y, z, poles, seed=7)
    assert a.shape == (len(adm.SCREEN_COLUMNS),)
    assert np.array_equal(a, adm.screen_scores(y, z, poles, seed=7))


def test_scan_row_agrees_with_the_dataset_solver():
    row = t1.scan(THETA[1])
    col = {name: j for j, name in enumerate(t1.SCAN_COLUMNS)}
    assert row[col["solved"]] == 1.0
    assert row[col["sigma"]] == pytest.approx(t1.solve(THETA[1])[2], abs=1e-9)
