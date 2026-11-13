"""Taking part of a Sobol dataset, and the admittance along the power axis."""

import numpy as np
import pytest

from sso_sciml import poweraxis
from sso_sciml.data import c1
from sso_sciml.sampling import part

COL = {name: j for j, name in enumerate(poweraxis.COLUMNS)}


def test_every_fourth_row_of_a_sobol_set_is_a_corner_of_it():
    # The reason `part` exists: a stride through the rows keeps positive power and strong
    # networks only.
    theta = c1.sample(4096, "pll")
    p = theta[:, c1.NAMES.index("p")]
    scr = theta[:, c1.NAMES.index("scr")]
    assert p[::4].min() >= 0.0
    assert p.min() < -0.99
    assert scr[::4].min() > 3.4
    assert scr.min() < 1.6
    # Without a phase-locked loop it keeps the weak networks and the light loads.
    theta = c1.sample(4096, "follow")
    assert theta[::4, c1.NAMES.index("scr")].max() < 3.5
    assert theta[::4, c1.NAMES.index("p")].min() > -0.6


def test_part_is_a_fair_and_repeatable_sample():
    theta = c1.sample(4096, "pll")
    rows = part(len(theta), 4)
    assert rows.size == 1024 and np.all(np.diff(rows) > 0)
    assert np.array_equal(rows, part(len(theta), 4))
    p = theta[rows, c1.NAMES.index("p")]
    scr = theta[rows, c1.NAMES.index("scr")]
    assert p.min() < -0.9 and p.max() > 0.9
    assert abs(np.mean(p < 0) - 0.5) < 0.05
    assert scr.min() < 1.6
    assert np.array_equal(part(10, 1), np.arange(10))


def test_lagrange_weights():
    nodes = np.array([0.1, 0.55, 1.0])
    assert poweraxis.weights(0.55, nodes) == pytest.approx([0.0, 1.0, 0.0])
    assert poweraxis.weights(0.3, nodes).sum() == pytest.approx(1.0)
    # Exact for a parabola.
    x = 0.37
    assert poweraxis.weights(x, nodes) @ nodes**2 == pytest.approx(x**2)


def test_scan_at_the_case_gives_its_growth_rate_and_three_scans_in_power_follow_it():
    rows = np.array([poweraxis.case(("pll", t)) for t in c1.sample(8, "pll")])
    assert rows.shape == (8, len(poweraxis.COLUMNS))
    err_exact = np.abs(rows[:, COL["exact"]] - rows[:, COL["true"]])
    err_three = np.abs(rows[:, COL["3 in power"]] - rows[:, COL["true"]])
    err_rated = np.abs(rows[:, COL["rated"]] - rows[:, COL["true"]])
    assert np.median(err_exact) < 1e-6
    assert np.median(err_three) < np.median(err_rated)
