"""A network with shunt capacitance on the plant bus."""

import numpy as np
import pytest

from sso_sciml import admittance as adm
from sso_sciml import hf, shunt
from sso_sciml import vectorfit as vf
from sso_sciml.data import c1

COL = {name: j for j, name in enumerate(shunt.COLUMNS)}
THETA = c1.sample(16, "pll")


def test_without_the_capacitor_it_is_the_network_of_the_earlier_steps():
    s = 2j * np.pi * np.array([-300.0, -5.0, 3.0, 40.0, 900.0])
    for theta in THETA[:6]:
        plant, net = hf.solve(theta, 0.0, 2e-4, 0.05, 0.1)
        same, _ = shunt.solve(theta, 0.0, 2e-4, 0.05, 0.1, 0.0)
        assert np.allclose(same.a, plant.a, atol=1e-9)
        assert np.allclose(shunt.network_impedance(net, s, 0.0), adm.network_impedance(net, s))
        old = vf.closed_loop_modes(plant.a, plant.b, plant.c, plant.d, net)
        new = shunt.closed_loop_modes(plant.a, plant.b, plant.c, plant.d, net, 0.0)
        assert new.size == old.size
        assert new.real.max() == pytest.approx(old.real.max(), abs=1e-6)


def test_modes_with_the_capacitor_are_zeros_of_the_loop_determinant():
    # Two independent routes: the state equations, and admittance times impedance.
    for theta in THETA[:6]:
        b_sh = 0.08
        plant, net = shunt.solve(theta, 0.0, 2e-4, 0.05, 0.1, b_sh)
        lam = shunt.closed_loop_modes(plant.a, plant.b, plant.c, plant.d, net, b_sh)
        plain = shunt.closed_loop_modes(plant.a, plant.b, plant.c, plant.d, net, 0.0)
        # The plant-bus voltage has become a state, and so has the transformer current,
        # which the filter capacitor branch lets differ from the line currents.
        assert lam.size == plain.size + 4
        top = lam[np.argsort(-lam.real)[:6]]

        def det(s, plant=plant, net=net, b_sh=b_sh):
            y, z = plant.admittance(s), shunt.network_impedance(net, s, b_sh)
            return np.linalg.det(np.eye(2) + y @ z)

        assert np.abs(det(top)).max() < 1e-6 * np.abs(det(top * 1.05 + 1.0)).min()


def test_steady_state_satisfies_the_network_with_the_capacitor():
    theta, b_sh = THETA[0], 0.2
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    net = c1.network(th)
    model = shunt.Converter(c1.controls(th, 0.0))
    v, i, _ = shunt.operating_point(model, th["p"], net, 0.0, 0.0, b_sh)
    assert abs(v) == pytest.approx(1.0)
    # Walk from the terminal to the infinite bus: it must come out at 1 pu, angle zero.
    v_bus = v - (net["r_t"] + 1j * net["x_t"]) * i
    i_grid = i - 1j * b_sh * v_bus
    line = net["r_e"] + 1j * (net["x_e"] - net["x_c"])
    path = net["r_p"] + 1j * net["x_p"]
    assert v_bus - i_grid * line * path / (line + path) == pytest.approx(1.0, abs=1e-8)


def test_case_row_repeats_step_28_without_the_capacitor():
    theta = THETA[2]
    row, old = shunt.case(("pll", theta)), hf.case(("pll", theta))
    assert row.shape == (len(shunt.COLUMNS),)
    old_col = {name: j for j, name in enumerate(hf.COLUMNS)}
    for v in hf.VARIANTS:
        assert row[COL[f"top_{v}"]] == pytest.approx(old[old_col[f"top_{v}"]], abs=1e-6)
    assert shunt.B_SH[0] <= row[COL["b_sh"]] <= shunt.B_SH[1]
    # A study that has the susceptance right gets the growth rate right.
    assert row[COL["ends_both"]] >= row[COL["given_both"]]
