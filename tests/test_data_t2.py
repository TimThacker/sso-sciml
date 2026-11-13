"""Dataset T2: the response in time (step 35)."""

import numpy as np
import torch

from sso_sciml.data import t2
from sso_sciml.physics.dfig import DFIG
from sso_sciml.solver.integrate import rk4

THETA = t2.sample(8)


def test_sample_is_fixed_and_in_range():
    assert np.array_equal(t2.sample(8), THETA)
    for j, par in enumerate(t2.SPEC):
        assert (THETA[:, j] >= par.low).all() and (THETA[:, j] <= par.high).all()
    high = t2.sample(8, "high_compensation")
    assert (high[:, 0] >= 0.50).all()
    assert np.abs(t2.scale_theta(THETA)).max() <= 1.0


def test_reduced_outputs_are_the_currents_of_the_reduced_model():
    model = DFIG()
    x = torch.linspace(0.1, 0.8, 8, dtype=torch.float64)
    p = model.params(0.1)
    c = model.currents(x, p)
    got = t2.reduced_outputs() @ x.numpy()
    assert np.allclose(got, [float(c["i_D"]), float(c["i_Q"]), 0.5, 0.6], atol=1e-12)


def test_reduced_response_is_the_time_domain_run_of_the_reduced_model():
    theta = THETA[0]
    th = dict(zip(t2.NAMES, map(float, theta), strict=True))
    from sso_sciml.physics.dfig import gains_from_bandwidth

    model = DFIG()
    kp, ki = gains_from_bandwidth(model.data, th["rsc_hz"])
    x0, p = model.equilibrium(th["k"] * model.data.x_e, w_r=th["w_r"], kp=kp, ki=ki)
    start = x0.clone()
    start[t2.VC] += t2.OFFSET
    _, x = rk4(model.rhs, start, p, 0.06, 1e-4, save_every=15)
    ref = (x - x0).numpy() @ t2.reduced_outputs().T
    got = t2.reduced_response(theta, np.arange(ref.shape[0]) * 1.5e-3)
    assert np.abs(got - ref).max() < 1e-6


def test_setup_and_a_short_run_on_the_cpu():
    rows = t2.setups(THETA[:3])
    assert all(r["solved"] == 1.0 for r in rows)
    assert rows[0]["reduced"].shape == (t2.N_T, 4)
    p = {k: np.array([r["p"][k] for r in rows]) for k in rows[0]["p"]}
    x_eq = np.array([r["x_eq"] for r in rows])
    states, obs, valid = t2.simulate(x_eq, p, device="cpu")
    assert states.shape == (3, t2.N_T, t2.N_STATE) and obs.shape == (3, t2.N_T, 5)
    assert ((valid > 0) & (valid <= t2.N_T)).all()
    assert np.isfinite(states).all()
    # At the first instant only the capacitor voltage is off its steady value.
    assert np.allclose(obs[:, 0], [0.0, 0.0, t2.OFFSET, 0.0, 0.0], atol=1e-6)
    assert np.allclose(states[:, 0, t2.VC], t2.OFFSET, atol=1e-7)
    # The reduced model starts the same way and stays near the full one at first.
    k = 4
    assert np.abs(obs[:, k, 2:4] - np.array([r["reduced"][k, 2:4] for r in rows])).max() < 2e-3
