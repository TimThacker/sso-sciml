"""Screens given network data with error."""

import numpy as np
import pytest

from sso_sciml import neterror
from sso_sciml.data import c1, t1

COL = {name: j for j, name in enumerate(neterror.COLUMNS)}
THETA = t1.sample(4)


def test_perturb_scales_each_quantity_by_its_factor():
    net = t1.network(THETA[0])
    wrong = neterror.perturb(net, np.array([1.1, 1.2, 1.3, 1.4, 1.5]))
    assert wrong["x_e"] == pytest.approx(1.1 * net["x_e"])
    assert wrong["r_e"] == pytest.approx(1.2 * net["r_e"])
    assert wrong["x_c"] == pytest.approx(1.3 * net["x_c"])
    assert wrong["x_p"] / wrong["r_p"] == pytest.approx(net["x_p"] / net["r_p"])
    assert wrong["x_t"] == pytest.approx(1.5 * net["x_t"])
    assert neterror.perturb(net, np.ones(5)) == pytest.approx(net)


def test_impedance_root_on_the_case_network_is_the_dataset_column():
    m = {name: j for j, name in enumerate(t1.MECHANISM_COLUMNS)}
    for theta in THETA:
        mech = t1.mechanism(theta)
        root = t1.impedance_root(theta, t1.network(theta))
        assert root == pytest.approx((mech[m["sigma_impedance"]], mech[m["f_impedance"]]))


def test_operating_point_on_a_given_network_matches_the_default():
    theta = THETA[1]
    _, x0, _ = t1.operating_point(theta)
    _, x1, _ = t1.operating_point(theta, t1.network(theta))
    assert np.allclose(x0.numpy(), x1.numpy(), atol=1e-12)


def test_a_wrong_network_moves_the_resonance_more_than_the_slow_mode():
    row = neterror.case(("t1", THETA[2]))
    assert row.shape == (len(neterror.COLUMNS),)
    capacitor = neterror.FACTORS.index("series capacitor")
    sub = abs(row[COL[f"sub_{capacitor}_up"]] - row[COL["true_sub"]])
    slow = abs(row[COL[f"slow_{capacitor}_up"]] - row[COL["true_slow"]])
    assert sub > 10 * slow


def test_converter_case_has_no_hand_formula_columns():
    row = neterror.case(("pll", c1.sample(2)[1]))
    assert np.isfinite(row[COL["true_sub"]]) or np.isfinite(row[COL["true_slow"]])
    assert np.isnan(row[COL["root_e0.02"]]) and np.isnan(row[COL["phasor_slow_e0.02"]])
