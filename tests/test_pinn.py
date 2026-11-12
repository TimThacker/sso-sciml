"""Phase 5 checks on the forward physics-informed network, on the CPU."""

import numpy as np
import pytest
import torch

from sso_sciml.models import pinn


@pytest.mark.parametrize("activation", ["tanh", "sin"])
@pytest.mark.parametrize("fourier", [(), (10.0, 40.0)])
def test_time_derivative_matches_automatic_differentiation(activation, fourier):
    torch.manual_seed(0)
    cfg = pinn.PinnConfig(0.2, width=16, depth=3, activation=activation, fourier_hz=fourier)
    net = pinn.TimeNet(3, 0.1, 0.2, cfg)
    win = pinn.Window(net, torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64),
                      torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64))  # fmt: skip
    t = torch.linspace(0.1, 0.3, 7, dtype=torch.float64).reshape(-1, 1)
    x, dx = win.with_rate(t)
    x_ref, dx_ref = torch.func.jvp(win, (t,), (torch.ones_like(t),))
    assert torch.allclose(x, x_ref, rtol=0, atol=1e-13)
    assert torch.allclose(dx, dx_ref, rtol=1e-11, atol=1e-11)


def test_initial_condition_is_exact():
    cfg = pinn.PinnConfig(1.0, width=8, depth=2)
    x_start = torch.tensor([0.3, -0.7], dtype=torch.float64)
    win = pinn.Window(pinn.TimeNet(2, 0.0, 1.0, cfg), x_start, torch.ones(2, dtype=torch.float64))
    assert torch.equal(win(torch.zeros(1, 1, dtype=torch.float64))[0], x_start)


def test_causal_weights_fall_only_after_a_large_earlier_residual():
    w = pinn.causal_weights(torch.tensor([0.0, 2.0, 0.0, 0.0], dtype=torch.float64), eps=1.0)
    assert w[0] == 1.0 and w[1] == 1.0
    assert w[2] == pytest.approx(np.exp(-2.0)) and w[3] == pytest.approx(np.exp(-2.0))


def _decay(x, p):
    return -p["a"] * x


@pytest.mark.parametrize("windows", [1, 2])
def test_learns_an_exponential_decay_from_the_equation_alone(windows):
    p = {"a": torch.tensor(3.0, dtype=torch.float64)}
    x0 = torch.tensor([1.0, -0.5], dtype=torch.float64)
    cfg = pinn.PinnConfig(1.0, steps=1500, n_col=128, width=32, depth=2, windows=windows)
    t = np.linspace(0.0, 1.0, 51)
    exact = x0.numpy() * np.exp(-3.0 * t)[:, None]

    def monitor(predict, t_reached):
        seen = t <= t_reached + 1e-12
        return np.abs(predict(t[seen]) - exact[seen]).max()

    predict, info = pinn.train_pinn(_decay, p, x0, torch.ones(2, dtype=torch.float64), cfg,
                                    device="cpu", monitor=monitor)  # fmt: skip
    assert np.abs(predict(t) - exact).max() < 0.02
    h = info["history"]
    assert h["step"][0] == 100 and h["step"][-1] == 1500
    assert h["residual"][-1] < h["residual"][0]
    assert h["monitor"][-1] < 0.02 and len(h["monitor"]) == len(h["step"])
    assert h["t_reached"][0] == pytest.approx(1.0 / windows) and h["t_reached"][-1] == 1.0


def test_trajectory_error_is_relative_to_the_deviation():
    x_eq, scale = np.zeros(2), np.array([1.0, 0.1])
    ref = np.array([[1.0, 0.1], [1.0, -0.1]])
    assert pinn.trajectory_error(ref, ref, x_eq, scale) == 0.0
    assert pinn.trajectory_error(np.zeros_like(ref), ref, x_eq, scale) == pytest.approx(1.0)
