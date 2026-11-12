"""One outage at a time."""

import numpy as np
import pytest

from sso_sciml import outage
from sso_sciml.data import c1, t1

COL = {name: j for j, name in enumerate(outage.COLUMNS)}
NAMES = list(outage.OUTAGES)


@pytest.fixture(scope="module")
def row():
    return outage.case(("t1", t1.sample(4)[2]))


def test_row_matches_its_columns(row):
    assert row.shape == (len(outage.COLUMNS),)
    assert all(row[COL[f"solved_{j}"]] == 1.0 for j in range(len(NAMES)))


def test_one_scan_follows_the_doubly_fed_plant_through_an_outage(row):
    # The steady state of the doubly-fed plant hardly moves with the network, so the
    # plant model taken before the outage gives the resonance after it.
    for j in range(len(NAMES)):
        assert row[COL[f"reuse_sub_{j}"]] == pytest.approx(row[COL[f"true_sub_{j}"]], abs=0.1)


def test_losing_the_parallel_path_hurts_and_losing_capacitor_helps(row):
    lost_path, half_capacitor = (
        NAMES.index("parallel path out"),
        NAMES.index("half the capacitor out"),
    )
    assert row[COL[f"true_sub_{lost_path}"]] > row[COL["intact_sub"]]
    assert row[COL[f"true_sub_{half_capacitor}"]] < row[COL["intact_sub"]]


def test_converter_on_a_given_network_matches_the_default():
    theta = c1.sample(2)[1]
    _, x0, _, net = c1.operating_point(theta, 0.0)
    _, x1, _, _ = c1.operating_point(theta, 0.0, net=net)
    assert np.allclose(x0.numpy(), x1.numpy(), atol=1e-10)
