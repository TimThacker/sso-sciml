"""A doubly-fed plant beside a converter (step 32)."""

import numpy as np
import pytest

from sso_sciml import admittance as adm
from sso_sciml import beside, multi
from sso_sciml import vectorfit as vf
from sso_sciml.data import c1, t1
from sso_sciml.physics.converter import Converter

THETA_M = t1.sample(8)
THETA_C = c1.sample(8, "pll")
S = 2j * np.pi * np.array([-45.0, -20.0, -3.0, 0.5, 8.0, 30.0, 200.0])


def _terminal(theta):
    model, x0, p = t1.operating_point(theta)
    vt = model.terminal_voltage(x0, p)
    return adm.DFIGScan(model, x0, p), complex(float(vt[0]), float(vt[1]))


def test_machine_scan_at_a_given_voltage_is_the_scan_of_the_dataset_case():
    # Same plant, same terminal voltage: the admittance must not depend on the network
    # the steady state came from. Also checks the turn into the common frame.
    for theta in THETA_M[:4]:
        ref, v = _terminal(theta)
        assert abs(np.angle(v)) > 0.05
        own = beside.machine(theta).scan(v)
        y = ref.admittance(S)
        assert np.abs(own.admittance(S) - y).max() < 1e-6 * np.abs(y).max()


def test_machine_alone_has_the_modes_of_the_full_model():
    for theta in THETA_M[:4]:
        lam = np.linalg.eigvals(t1.state_matrix(theta))
        models, shares, _ = beside.scans(beside.machine(theta), None, 1.0, t1.network(theta), 0.0)
        got = beside._together(models, shares, t1.network(theta), 0.0)
        assert got.real.max() == pytest.approx(lam.real.max(), abs=1e-6)


def test_converter_alone_has_the_modes_of_the_full_model():
    for theta in THETA_C[:4]:
        model, x0, p, net = c1.operating_point(theta, 0.0)
        conv = (Converter(c1.controls(dict(zip(c1.NAMES, theta, strict=True)), 0.0)),
                float(theta[0]))  # fmt: skip
        models, shares, _ = beside.scans(None, conv, 0.0, net, 0.0)
        got = beside._together(models, shares, net, 0.0)
        assert got.real.max() == pytest.approx(c1.growth(model, x0, p)[0], abs=1e-6)


def test_units_at_one_terminal_are_the_units_side_by_side():
    job = multi.sample(4, "pll")[1]
    _, theta_s, theta_l, share = job
    ths = dict(zip(c1.NAMES, map(float, theta_s), strict=True))
    thl = dict(zip(c1.NAMES, map(float, theta_l), strict=True))
    net = c1.network(ths)
    units = [multi.Unit(Converter(c1.controls(ths, 0.0)), abs(ths["p"]), share, regulate=True),
             multi.Unit(Converter(c1.controls(thl, 0.0)), -abs(thl["p"]), 1 - share)]  # fmt: skip
    scans = multi.plant(units, *multi.operating_point(units, net))
    a, b, c = multi.side_by_side(scans, [share, 1 - share])
    ref = vf.closed_loop_modes(a, b, c, np.zeros((2, 2)), net)
    got = beside._together(scans, [share, 1 - share], net, 0.0)
    assert got.size == ref.size
    assert np.sort(got.real)[-6:] == pytest.approx(np.sort(ref.real)[-6:], abs=1e-6)


def test_own_transformer_of_a_single_unit_is_the_common_one():
    theta = THETA_C[2]
    net = c1.network(dict(zip(c1.NAMES, map(float, theta), strict=True)))
    conv = (Converter(c1.controls(dict(zip(c1.NAMES, theta, strict=True)), 0.0)), float(theta[0]))
    top = []
    for name in beside.LAYOUTS:
        lay_net, z_own = beside.layout(net, name)
        models, shares, _ = beside.scans(None, conv, 0.0, lay_net, z_own)
        top.append(beside._together(models, shares, lay_net, z_own).real.max())
    assert top[1] == pytest.approx(top[0], abs=1e-6)


def test_case_row():
    row = beside.case(beside.sample(4, "pll")[0])
    col = {name: j for j, name in enumerate(beside.COLUMNS)}
    assert row.shape == (len(beside.COLUMNS),)
    assert row[col["solved_bus"]] == 1.0
    assert np.isfinite(row[col["top_bus"]]) and np.isfinite(row[col["top_apart"]])
