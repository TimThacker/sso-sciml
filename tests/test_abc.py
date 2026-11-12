"""Phase 2 checks on the phase-coordinate implementation.

Each test pins a number from docs/steps/07-phase-coordinates.md.
"""

import numpy as np
import pytest

from sso_sciml.physics.dfig import DFIG, gains_from_bandwidth
from sso_sciml.physics.dfig_abc import PhaseModel, to_dq, to_phase, trapezoidal
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole


@pytest.fixture(scope="module")
def event():
    m = DFIG()
    kp, ki = gains_from_bandwidth(m.data, 10.0)
    x_a, p_a = m.equilibrium(0.40 * m.data.x_e, w_r=1.0, kp=kp, ki=ki)
    p_b = {**p_a, "x_c": p_a["x_c"] * 1.25}
    return m, x_a, p_b


def test_frame_transforms_are_inverses():
    for theta in (0.0, 0.7, 4.0):
        assert np.allclose(to_dq(theta) @ to_phase(theta), np.eye(2), atol=1e-14)


def test_steady_state_stays_steady_in_phase_coordinates():
    m = DFIG()
    kp, ki = gains_from_bandwidth(m.data, 3.0)
    x_a, p_a = m.equilibrium(0.40 * m.data.x_e, w_r=1.0, kp=kp, ki=ki)
    pm = PhaseModel.from_params(p_a, m.data.wb)
    t, x = trapezoidal(pm, PhaseModel.from_dq_state(x_a.numpy()), 0.1, 5e-5, 20)
    back = pm.to_dq_state(x, t)
    # not exact: the trapezoidal rule misplaces the 60 Hz forcing by a term in dt squared
    assert np.abs(back - x_a.numpy()).max() < 6e-4
    assert np.abs(x[:, 0:3].sum(axis=1)).max() < 1e-12  # no zero sequence


def test_phase_model_follows_the_dq_model_and_converges_at_second_order(event):
    m, x_a, p_b = event
    _, ref = rk4(m.rhs, x_a, p_b, 0.3, 2.5e-5, save_every=40)
    ref = ref.numpy()
    err = []
    for dt in (5e-5, 1e-4):
        pm = PhaseModel.from_params(p_b, m.data.wb)
        t, x = trapezoidal(
            pm, PhaseModel.from_dq_state(x_a.numpy()), 0.3, dt, int(round(1e-3 / dt))
        )
        err.append(np.abs(pm.to_dq_state(x, t)[:, :6] - ref[:, :6]).max())
    assert err[0] / np.abs(ref - ref[0]).max() < 3e-3
    assert abs(np.log2(err[1] / err[0]) - 2.0) < 0.05


def test_growth_rate_from_phase_current_matches_the_eigenvalue(event):
    m, x_a, p_b = event
    md = modes(jacobian(m.rhs, x_a, p_b))
    lam = md.eigenvalues[md.sigma.argmax()]
    assert abs(lam.real - 8.637) < 2e-3
    pm = PhaseModel.from_params(p_b, m.data.wb)
    t, x = trapezoidal(pm, PhaseModel.from_dq_state(x_a.numpy()), 0.6, 5e-5, 20)
    f_sub = 60.0 - abs(lam.imag) / (2 * np.pi)
    pole = nearest_pole(matrix_pencil(pm.stator_current(x)[t > 0.1, 0], 1e-3, 12), f_sub)
    assert abs(pole.imag / (2 * np.pi) - f_sub) < 1e-3  # 34.665 Hz in the phase current
    assert abs(pole.real - lam.real) < 1e-3
