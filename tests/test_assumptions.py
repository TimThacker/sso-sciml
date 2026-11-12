"""Phase 2 checks on the assumption studies.

Each test pins a number from docs/steps/06-assumption-studies.md.
"""

import numpy as np
import pytest

from sso_sciml.physics.dfig_full import DFIGFull
from sso_sciml.physics.dfig_meshed import DFIGMeshed
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes, quasi_static
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

FAST = [0, 1, 4, 5, 6, 7, 23, 24]


@pytest.fixture(scope="module")
def case():
    m = DFIGMeshed()
    x0, p = m.equilibrium(0.60 * m.data.x_e, w_r=0.75)
    a = jacobian(m.rhs, x0, p).numpy()
    return m, x0, p, a


def test_quasi_static_reduction_of_a_two_state_system():
    a = np.array([[-1.0, 2.0], [3.0, -100.0]])
    red, slow = quasi_static(a, [1])
    assert slow == [0]
    assert abs(red[0, 0] - (-1.0 + 2.0 * 3.0 / 100.0)) < 1e-14


def test_dropping_transients_removes_the_unstable_mode(case):
    _, _, _, a = case
    full = np.linalg.eigvals(a)
    red = np.linalg.eigvals(quasi_static(a, FAST)[0])
    assert abs(full.real.max() - 1.616) < 2e-3
    assert red.size == 17
    assert abs(red.real.max() + 0.196) < 2e-3


def test_dropping_transients_keeps_the_slow_modes(case):
    _, _, _, a = case
    full = np.linalg.eigvals(a)
    red = np.linalg.eigvals(quasi_static(a, FAST)[0])
    slow = full[np.abs(full) < 2 * np.pi * 5.0]
    assert slow.size >= 5
    assert max(np.min(np.abs(red - lam)) for lam in slow) < 0.6


def test_quasi_static_model_reports_every_radial_case_stable():
    m = DFIGFull()
    for k, w in ((0.5, 0.75), (0.9, 1.0), (0.9, 1.2)):
        x0, p = m.equilibrium(k * m.data.x_e, w_r=w)
        a = jacobian(m.rhs, x0, p).numpy()
        assert np.linalg.eigvals(a).real.max() > 3.0
        assert np.linalg.eigvals(quasi_static(a, FAST[:6])[0]).real.max() < 0.0


def test_capacitor_as_a_fixed_reactance_puts_the_mode_at_the_wrong_frequency():
    m = DFIGFull()
    x0, p = m.equilibrium(0.5 * m.data.x_e, w_r=0.75)
    a = jacobian(m.rhs, x0, p).numpy()
    lam = np.linalg.eigvals(quasi_static(a, [4, 5])[0])
    worst = lam[lam.real.argmax()]
    assert abs(worst.real - 4.15) < 0.02
    assert abs(abs(worst.imag) / (2 * np.pi) - 39.5) < 0.1  # the full model has 26.1 Hz


def test_rk4_growth_rate_error_at_one_millisecond(case):
    m, x0, p, a = case
    md = modes(a)
    lam = md.eigenvalues[md.sigma.argmax()]
    z = lam * 1e-3
    predicted = (np.log(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24) / 1e-3).real
    assert abs((predicted - lam.real) + 7.6e-5) < 5e-6
    x = x0.clone()
    x[4] += 1e-4
    t, traj = rk4(m.rhs, x, p, 1.0, 1e-3)
    wave = m.outputs(traj, p)["p_t"].numpy() - float(m.outputs(x0, p)["p_t"])
    pole = nearest_pole(matrix_pencil(wave[t.numpy() > 0.3], 1e-3, 10), abs(lam.imag) / 6.2832)
    assert abs(pole.real - predicted) < 1e-5


def test_trapezoidal_rule_needs_a_smaller_step_than_rk4(case):
    _, _, _, a = case
    md = modes(a)
    lam = md.eigenvalues[md.sigma.argmax()]
    z = lam * 1e-3
    trap = (np.log((1 + z / 2) / (1 - z / 2)) / 1e-3).real - lam.real
    assert abs(trap + 0.0078) < 2e-4
