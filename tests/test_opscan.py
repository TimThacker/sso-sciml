"""The admittance of a converter as a function of its operating point."""

import numpy as np
import pytest

from sso_sciml import opscan
from sso_sciml.data import c1

COL = {name: j for j, name in enumerate(opscan.COLUMNS)}
N_OUT = len(opscan.OUTAGES)


@pytest.fixture(scope="module")
def rows():
    theta = c1.sample(6)
    return np.array([opscan.case(("pll", t)) for t in theta])


def test_row_matches_its_columns(rows):
    assert rows.shape == (6, len(opscan.COLUMNS))


def test_scan_at_the_operating_point_of_the_case_is_the_dataset_scan():
    theta = c1.sample(6)[3]
    model, x0, p, _ = c1.operating_point(theta, 0.0)
    y_case = c1.scan_pair(theta, 0.0)[0][:: opscan.adm.NOISE_STEP]
    y = opscan.scan_at(model, p, float(theta[0]), float(p["iq_ref"]), 0.0)
    # The two differ by the rotation that puts the terminal voltage on the real axis,
    # which leaves the eigenvalues of Y unchanged.
    assert np.allclose(
        np.sort_complex(np.linalg.eigvals(y)), np.sort_complex(np.linalg.eigvals(y_case)), atol=1e-8
    )


def test_a_scan_retaken_after_the_outage_gives_the_growth_rate(rows):
    err = []
    for j in range(N_OUT):
        solved = rows[:, COL[f"solved_{j}"]] > 0
        assert solved.any()
        err.append(np.abs(rows[solved, COL[f"exact_{j}"]] - rows[solved, COL[f"true_{j}"]]))
    err = np.concatenate(err)
    # Exact to rounding for most; a well-damped mode far from the axis is read less
    # finely from the scan.
    assert np.median(err) < 1e-6
    assert err.max() < 0.1


def test_a_parabola_through_three_scans_follows_the_operating_point():
    theta = c1.sample(6)[1]
    model, _, p, _ = c1.operating_point(theta, 0.0)
    p_src, u0, u1 = float(theta[0]), float(p["iq_ref"]), float(p["iq_ref"]) - 0.13

    def scan(u):
        return opscan.scan_at(model, p, p_src, u, 0.0)

    y0, y_lo, y_hi, target = scan(u0), scan(u0 - opscan.STEP), scan(u0 + opscan.STEP), scan(u1)
    d = (u1 - u0) / opscan.STEP
    parabola = y0 + 0.5 * d * (y_hi - y_lo) + 0.5 * d * d * (y_hi - 2 * y0 + y_lo)
    assert np.abs(parabola - target).max() < 0.1 * np.abs(y0 - target).max()


def test_between_scans_is_exact_at_a_scan_and_held_outside_the_range():
    grid = np.array([0.0, 1.0, 2.0])
    scans = [np.full((2, 2), v) for v in (10.0, 20.0, 40.0)]
    assert np.allclose(opscan._between(grid, scans, 1.0), 20.0)
    assert np.allclose(opscan._between(grid, scans, 1.5), 30.0)
    assert np.allclose(opscan._between(grid, scans, 5.0), 40.0)
