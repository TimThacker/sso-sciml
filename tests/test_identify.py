"""Identification through the simulator and the waveform network (step 39). CPU."""

import numpy as np
import pytest

from sso_sciml import identify as idn
from sso_sciml.data import t2
from sso_sciml.models import t3


def test_unscale_inverts_the_scaling():
    theta = t2.sample(16)
    assert np.allclose(idn.unscale(t2.scale_theta(theta)), theta)


@pytest.fixture(scope="module")
def stable():
    """Two cases at low compensation, which decay and do not trip."""
    return np.array([[-0.8, 0.1, 0.0, 0.2], [-0.6, -0.3, 0.3, -0.2]])


def test_jacobian_predicts_a_small_change(stable):
    jac, ok = idn.jacobian(stable, device="cpu", processes=1)
    assert ok.all() and jac.shape[0] == 2 and jac.shape[2] == 4
    du = np.array([0.002, -0.0015, 0.0025, 0.001])
    y0, _ = idn.waveform(stable, "cpu", 1)
    y1, _ = idn.waveform(stable + du, "cpu", 1)
    lin = jac @ du
    assert np.abs(y1 - y0 - lin).max() < 0.05 * np.abs(lin).max()


def test_bound_shrinks_with_the_noise(stable):
    jac, _ = idn.jacobian(stable[:1], device="cpu", processes=1)
    a, b = idn.bound(jac, np.array([1e-3])), idn.bound(jac, np.array([1e-4]))
    assert np.allclose(a / b, 10.0) and (a > 0).all()


def test_gauss_newton_finds_the_parameters_of_a_clean_record(stable):
    y, _ = idn.waveform(stable[:1], "cpu", 1)
    start = stable[:1] + np.array([[0.02, -0.04, 0.06, 0.05]])
    u, history = idn.gauss_newton(y, start, iterations=6, device="cpu", processes=1)
    assert history[-1, 0] < 0.02 * history[0, 0]
    assert np.abs(u - stable[:1])[0, :3].max() < 0.02


def test_wave_network_trains_and_decodes():
    rng = np.random.default_rng(0)
    t = np.arange(577) / 1920
    f = rng.uniform(25, 45, 64)
    sigma = rng.uniform(-5, 10, 64)
    clean = np.exp(sigma[:, None] * t) * np.cos(2 * np.pi * (60 - f[:, None]) * t) * 0.02
    target = t3.targets(sigma, f, rng.uniform(-1, 1, (64, 4)))
    cfg = t3.Config(steps=20, batch=16, width=8, eval_every=10)
    predict, info = t3.train(clean[:48], target[:48], (clean[48:], target[48:]), 0, cfg, "cpu")
    out = predict(clean)
    assert out["sigma"].shape == (64,) and out["u"].shape == (64, 4)
    assert np.isfinite(out["f"]).all() and len(info["history"]["step"]) == 2
