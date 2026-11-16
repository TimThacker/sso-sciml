"""The full controls written a second time in phase coordinates (step 44). CPU."""

import numpy as np
import pytest

from sso_sciml.data import t2
from sso_sciml.physics import dfig_full_abc as fa
from sso_sciml.solver.integrate import rk4


@pytest.fixture(scope="module")
def case():
    model, x_c, w_r = t2.plant(np.array([0.4, 1.1, 15.0, 15.0]))
    x0, p = model.equilibrium(x_c, w_r=w_r)
    return model, x0, p, fa.PhaseFull(p, model.data.wb)


def test_frames_are_inverse():
    for angle in (0.0, 0.7, -2.1):
        assert np.allclose(fa.to_dq(angle) @ fa.to_phase(angle), np.eye(2))


def test_steady_state_of_the_first_model_is_a_steady_rotation_of_the_second(case):
    model, x0, p, pm = case
    z0 = pm.from_full_state(x0.numpy())
    rate = pm.rhs(0.0, z0)
    wb, h = model.data.wb, 1e-7
    turn = (fa.to_phase(wb * h) - fa.to_phase(-wb * h)) / (2 * h)
    for s in range(4):
        sl = slice(3 * s, 3 * s + 3)
        assert np.allclose(rate[sl], turn @ (fa.to_dq(0.0) @ z0[sl]), rtol=1e-7, atol=1e-6)
    assert rate[13] == pytest.approx(wb)
    assert np.abs(rate[[12, 14, *range(15, 27)]]).max() < 1e-9


def test_trajectories_agree_after_an_offset(case):
    model, x0, p, pm = case
    start = x0.clone()
    start[4] += 1e-3
    t, x = rk4(model.rhs, start, p, 0.05, 5e-5, save_every=20)
    t = t.numpy()
    z = fa.simulate(pm, pm.from_full_state(start.numpy()), t)
    rest = fa.to_dq(0.0) @ pm.from_full_state(x0.numpy())[9:12]
    vc_a = (x[:, 4:6] - x0[4:6]).numpy()
    vc_b = pm.capacitor_dq(t, z) - rest
    assert np.abs(vc_b - vc_a).max() < 1e-6 * np.abs(vc_a).max() + 1e-12
    assert np.abs(z[:, 23] - x[:, 19].numpy()).max() < 1e-8
