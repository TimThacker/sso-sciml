"""System C: generic converter, its admittance, and the screens built on scans."""

import numpy as np
import pytest
import torch

from sso_sciml import admittance as adm
from sso_sciml.data import c1
from sso_sciml.physics.converter import WB, Converter, ConverterControls, active_states
from sso_sciml.solver.linearize import jacobian

NET = {"r_e": 0.02, "x_e": 0.4, "x_c": 0.16, "r_p": 0.04, "x_p": 0.8, "r_t": 0.007, "x_t": 0.1}
GRID = adm.scan_grid()
S = 2j * np.pi * GRID


def _case(p_src, **controls):
    model = Converter(ConverterControls(**controls))
    x0, p = model.equilibrium(p_src, **NET)
    vt = model.terminal_voltage(x0, p)
    v_t = complex(float(vt[0]), float(vt[1]))
    i = complex(float(x0[0] + x0[2]), float(x0[1] + x0[3]))
    return model, x0, p, v_t, i


def _modes(model, x0, p):
    keep = active_states(model.controls.follow)
    return np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy()[np.ix_(keep, keep)])


@pytest.mark.parametrize(("p_src", "follow"), [(0.8, 0.0), (-0.8, 0.0), (-0.5, 1.0)])
def test_steady_state_holds_and_carries_the_power(p_src, follow):
    model, x0, p, v_t, i = _case(p_src, follow=follow, t_ff=0.0002 if follow else 0.002)
    assert float(model.rhs(x0, p).abs().max()) < 1e-9
    r_f, x_f = model.controls.r_f, model.controls.x_f
    assert ((v_t + (r_f + 1j * x_f) * i) * np.conj(i)).real == pytest.approx(p_src, abs=1e-9)
    if not follow:
        assert abs(v_t) == pytest.approx(1.0, abs=1e-9)


def test_converter_alone_has_the_same_steady_state():
    model, x0, p, v_t, i = _case(0.6)
    z0 = model.terminal_equilibrium(v_t, i)
    v0 = torch.tensor([v_t.real, v_t.imag], dtype=torch.float64)
    assert float(model.rhs_terminal(z0, v0, p).abs().max()) < 1e-9
    assert torch.allclose(z0[2:], x0[6:], atol=1e-12)


def test_network_impedance_at_zero_is_the_60_hz_impedance():
    z = adm.network_impedance(NET, np.array([0.0]))[0]
    line = NET["r_e"] + 1j * (NET["x_e"] - NET["x_c"])
    path = NET["r_p"] + 1j * NET["x_p"]
    z60 = NET["r_t"] + 1j * NET["x_t"] + line * path / (line + path)
    assert z[0, 0] == pytest.approx(z60.real)
    assert z[1, 0] == pytest.approx(z60.imag)
    assert z[0, 1] == pytest.approx(-z60.imag)


@pytest.mark.parametrize(("p_src", "follow"), [(0.9, 0.0), (-0.9, 0.0), (-0.6, 1.0)])
def test_modes_of_the_full_model_are_roots_of_the_scan_determinant(p_src, follow):
    """Two formulations: state matrix of the whole system, and Y(s) of the converter
    alone against Z(s) of the network."""
    model, x0, p, v_t, i = _case(p_src, follow=follow, t_ff=0.0002 if follow else 0.002)
    lam = _modes(model, x0, p)
    lam = lam[np.argsort(-lam.real)[:6]]
    scan = adm.ConverterScan(model, v_t, i, p)
    det = np.linalg.det(np.eye(2) + scan.admittance(lam) @ adm.network_impedance(NET, lam))
    assert np.abs(det).max() < 1e-8


def test_nyquist_count_matches_the_eigenvalues_on_a_stable_and_an_unstable_case():
    for pll_hz, expect_unstable in ((5.0, False), (60.0, True)):
        model = Converter(ConverterControls(pll_hz=pll_hz, ff=0.0))
        net = {**NET, "x_e": 1.0, "x_p": 2.0, "r_e": 0.05, "r_p": 0.1, "x_c": 0.0}
        x0, p = model.equilibrium(0.9, **net)
        vt = model.terminal_voltage(x0, p)
        v_t = complex(float(vt[0]), float(vt[1]))
        i = complex(float(x0[0] + x0[2]), float(x0[1] + x0[3]))
        n_eig = int((_modes(model, x0, p).real > 1e-9).sum())
        scan = adm.ConverterScan(model, v_t, i, p)
        n_scan = adm.unstable_modes(scan.admittance(S), adm.network_impedance(net, S))
        assert (n_eig > 0) == expect_unstable
        assert n_scan == n_eig


def test_current_loop_alone_is_passive():
    """With the PLL and the DC voltage loop nearly frozen and no feed-forward the
    converter is a resistor and a reactor behind a voltage: passive at every frequency."""
    model, _, p, v_t, i = _case(0.8, pll_hz=1e-3, dc_hz=1e-3, ff=0.0)
    y = adm.ConverterScan(model, v_t, i, p).admittance(S[np.abs(GRID) > 1.0])
    assert adm.passivity_index(y).min() > 0


def test_constant_power_shows_as_negative_conductance_for_a_load_only():
    """Inside the DC voltage loop a load at power P and voltage V has conductance near
    -P / V^2 along the voltage. A source at the same power has a positive one."""
    low = np.array([2j * np.pi * 0.5])
    g = {}
    for p_src in (-0.8, 0.8):
        model, _, p, v_t, i = _case(p_src, pll_hz=1e-3, dc_hz=20.0, ff=0.0)
        y = adm.ConverterScan(model, v_t, i, p).admittance(low)[0]
        unit = np.array([v_t.real, v_t.imag]) / abs(v_t)
        g[p_src] = float((unit @ y @ unit).real)
    assert g[-0.8] == pytest.approx(-0.8, rel=0.1)
    assert g[0.8] == pytest.approx(0.8, rel=0.1)


def test_small_gain_score_below_one_only_on_stable_cases():
    theta = c1.sample(64, "pll")
    out = np.array([c1.evaluate(row, 0.0) for row in theta])
    col = {name: out[:, j] for j, name in enumerate(c1.COLUMNS)}
    ok = col["solved"] > 0
    assert ok.sum() > 56
    assert (col["sigma"][ok & (col["small_gain"] < 1.0)] < 0).all()
    assert ((col["count"] > 0) == (col["sigma"] > 0))[ok].all()


def test_sampling_is_reproducible_and_families_differ():
    a, b = c1.sample(64, "follow"), c1.sample(64, "follow")
    assert np.array_equal(a, b)
    assert (a[:, 0] <= -0.1).all()
    assert not np.array_equal(a[:, 1], c1.sample(64, "pll")[:, 1])
    assert WB == pytest.approx(376.991, abs=1e-3)
