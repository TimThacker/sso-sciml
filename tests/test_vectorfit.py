"""Least-squares rational fit of a scan, and a fitted plant closed on its network."""

import numpy as np
import pytest

from sso_sciml import admittance as adm
from sso_sciml import scanfit
from sso_sciml import vectorfit as vf
from sso_sciml.data import c1, t1
from sso_sciml.solver.linearize import jacobian

THETA = t1.sample(3)
GRID = adm.scan_grid()
HALF = GRID >= 0


def full_modes(theta):
    model, x0, p = t1.operating_point(theta)
    return model, x0, p, np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy())


def test_plant_closed_on_its_network_has_the_modes_of_the_full_model():
    for theta in THETA:
        model, x0, p, lam = full_modes(theta)
        plant = adm.DFIGScan(model, x0, p)
        modes = vf.closed_loop_modes(plant.a, plant.b, plant.c, np.zeros((2, 2)), t1.network(theta))
        assert modes.size == lam.size
        for value in lam:
            assert np.abs(modes - value).min() < 1e-6 * max(1.0, abs(value))


def test_closed_loop_without_series_capacitor():
    theta = c1.sample(1)[0].copy()
    theta[c1.NAMES.index("k")] = 0.0
    model, x0, p, net = c1.operating_point(theta, 0.0)
    lam = np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy())
    vt = model.terminal_voltage(x0, p)
    i = complex(float(x0[0] + x0[2]), float(x0[1] + x0[3]))
    plant = adm.ConverterScan(model, complex(float(vt[0]), float(vt[1])), i, p)
    modes = vf.closed_loop_modes(plant.a, plant.b, plant.c, np.zeros((2, 2)), net)
    # With no capacitor the full model still carries its two idle states, at 60 Hz.
    lam = lam[np.abs(np.abs(lam.imag) - 2 * np.pi * 60.0) + np.abs(lam.real) > 1e-6]
    assert modes.size == lam.size
    assert modes.real.max() == pytest.approx(lam.real.max(), abs=1e-6)


def test_vector_fit_recovers_a_known_rational_function():
    s = 2j * np.pi * np.geomspace(0.1, 1000.0, 120)
    poles = np.array([-3.0 + 40.0j, -50.0 + 900.0j])
    y = sum(
        r / (s - a) + np.conj(r) / (s - np.conj(a))
        for a, r in zip(poles, (2 + 1j, 30 - 5j), strict=True)
    )
    fit = vf.vector_fit(s, (y + 0.7)[:, None], pairs=2)
    assert fit["residual"] < 1e-9
    found = fit["poles"][np.argsort(fit["poles"].imag)]
    assert np.allclose(found, poles, rtol=1e-8)


def test_fit_of_a_clean_scan_gives_the_growth_rate():
    for theta in THETA:
        _, _, _, lam = full_modes(theta)
        y, _, _ = t1.scan_pair(theta)
        fit = vf.fit_admittance(GRID[HALF][::16], y[HALF][::16], noise=0.0)
        modes = vf.closed_loop_modes(fit["a"], fit["b"], fit["c"], fit["d"], t1.network(theta))
        assert max(scanfit.band_growth(modes)) == pytest.approx(lam.real.max(), abs=1e-5)
        # Fewer states than two per fitted pole: the surplus was cut. How many survive
        # on a clean scan depends on rounding, so the count is not pinned to the 21 of
        # the plant.
        assert fit["states"] < 4 * fit["pairs"]


def test_band_growth_sorts_modes_by_frequency():
    w = 2 * np.pi
    lam = np.array([-0.3 + 0.5j * w, 4.0 + 20j * w, -9.0 + 300j * w])
    assert scanfit.band_growth(lam) == pytest.approx([-0.3, 4.0, -9.0])
    assert scanfit.band_growth(lam[:1])[1] == float("-inf")


def test_case_row_matches_its_columns_and_the_full_model():
    row = scanfit.case(("t1", THETA[0]))
    assert row.shape == (len(scanfit.COLUMNS),)
    lam = full_modes(THETA[0])[3]
    assert max(row[:3]) == pytest.approx(lam.real.max(), abs=1e-9)
