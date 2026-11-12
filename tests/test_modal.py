"""Checks on the response computed from the modes of the state matrix."""

from pathlib import Path

import numpy as np
import scipy.linalg

from sso_sciml.data import t1
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian
from sso_sciml.solver.modal import linear_response

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "t1_sample.npz"


def test_linear_response_is_the_matrix_exponential():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5)) * 3.0
    dx0 = rng.standard_normal(5)
    t = np.array([0.0, 0.1, 0.7])
    got = linear_response(a, dx0, t)
    want = np.array([scipy.linalg.expm(a * ti) @ dx0 for ti in t])
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9)
    assert np.allclose(got[0], dx0)


def test_operating_point_is_a_steady_state_of_the_full_model():
    z = np.load(SAMPLE)
    model, x0, p = t1.operating_point(z["theta"][0])
    assert x0.shape == (t1.N_STATE,)
    assert float(model.rhs(x0, p).abs().max()) < 1e-8
    assert np.array_equal(jacobian(model.rhs, x0, p).numpy(), t1.state_matrix(z["theta"][0]))


def test_small_offset_response_matches_time_stepping_on_the_full_model():
    z = np.load(SAMPLE)
    model, x_eq, p = t1.operating_point(z["theta"][0])
    a = jacobian(model.rhs, x_eq, p).numpy()
    x0 = x_eq.clone()
    x0[4] += 0.01
    t, x = rk4(model.rhs, x0, p, 0.05, 2.5e-4, save_every=4)
    dev = x.numpy() - x_eq.numpy()
    lin = linear_response(a, (x0 - x_eq).numpy(), t.numpy())
    electrical = [0, 1, 2, 3, 4, 5, 6, 7, 23, 24]
    err = np.linalg.norm(lin[:, electrical] - dev[:, electrical])
    assert err / np.linalg.norm(dev[:, electrical]) < 5e-3
