"""Phase 2 checks on the doubly-fed machine with full controls (System B2, stage 2).

Each test pins a number from docs/steps/04-dfig-full-controls.md.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from sso_sciml.physics.dfig import DFIG, gains_from_bandwidth
from sso_sciml.physics.dfig_full import N_STATE, STATE_NAMES, W_R, Controls, DFIGFull
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes, newton_equilibrium
from sso_sciml.solver.modal import matrix_pencil, nearest_pole


@pytest.fixture(scope="module")
def model() -> DFIGFull:
    return DFIGFull()


def _sso(model, k, w_r, **overrides):
    x0, p = model.equilibrium(k * model.data.x_e, w_r=w_r, **overrides)
    md = modes(jacobian(model.rhs, x0, p))
    cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 58.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    return md, j, x0, p


# ------------------------------------------------------------------ steady state
def test_state_vector_layout():
    assert len(STATE_NAMES) == N_STATE == 23
    assert STATE_NAMES[W_R] == "w_r"


@pytest.mark.parametrize("w_r", [0.75, 1.0, 1.2])
def test_equilibrium_is_a_fixed_point_on_the_power_curve(model, w_r):
    x0, p = model.equilibrium(0.5 * model.data.x_e, w_r=w_r)
    assert model.rhs(x0, p).abs().max() < 1e-11
    out = model.outputs(x0, p)
    assert abs(float(out["p_t"]) - (w_r / 1.2) ** 3) < 1e-10
    assert abs(float(out["q_t"])) < 1e-10
    # Rotor power flows into the rotor below synchronous speed and out of it above. At
    # synchronous speed the converter supplies only the rotor copper loss.
    if w_r != 1.0:
        assert np.sign(float(out["p_rsc"])) == np.sign(w_r - 1.0)
    else:
        assert -0.01 < float(out["p_rsc"]) < 0
    # DC link balanced, mechanical power covers terminal power plus copper losses.
    assert abs(float(out["p_rsc"] - out["p_gsc"])) < 1e-12
    assert float(p["p_m"]) > float(out["p_t"])


def test_newton_returns_to_the_same_equilibrium(model):
    x0, p = model.equilibrium(0.2 * model.data.x_e, w_r=1.2)  # a stable case
    start = x0 + 1e-4 * torch.linspace(-1, 1, N_STATE, dtype=torch.float64)
    assert (newton_equilibrium(model.rhs, start, p) - x0).abs().max() < 1e-8


def test_rhs_is_differentiable_in_controller_settings(model):
    x0, p = model.equilibrium(0.5 * model.data.x_e, w_r=1.0)
    for key in ("kp", "pll_kp", "t_ff", "x_f"):
        v = p[key].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(model.rhs(x0 + 1e-3, {**p, key: v}).pow(2).sum(), v)
        assert torch.isfinite(g) and g.abs() > 0


# ------------------------------------------------------------------ against stage 1
def test_all_additions_off_reproduces_the_current_loop_model(model):
    off = DFIGFull(model.data, replace(Controls(), pll_hz=1e-3, outer_hz=1e-3, x_f=1e3, dc_hz=1e-3))
    s1 = DFIG()
    kp, ki = gains_from_bandwidth(model.data, 10.0)
    for k, w_r in ((0.2, 0.75), (0.5, 1.0), (0.8, 1.2)):
        md, j, _, _ = _sso(off, k, w_r, h=1e6)
        p = s1.params(k * model.data.x_e, w_r=w_r, kp=kp, ki=ki)
        lam = np.linalg.eigvals(jacobian(s1.rhs, torch.zeros(8, dtype=torch.float64), p).numpy())
        f = np.abs(lam.imag) / (2 * np.pi)
        assert abs(md.sigma[j] - lam.real[(f > 2.0) & (f < 58.0)].max()) < 2e-3


def test_growth_rates_with_and_without_feed_forward(model):
    no_ff = DFIGFull(model.data, replace(Controls(), ff=0.0))
    got = [float(_sso(m, 0.5, w)[0].sigma[_sso(m, 0.5, w)[1]])
           for m in (model, no_ff) for w in (0.75, 1.0, 1.2)]  # fmt: skip
    assert np.allclose(got, [17.407, 8.525, 3.561, 10.051, 2.540, -1.817], atol=3e-3)


def test_slow_feed_forward_filter_damps_the_mode(model):
    sig = []
    for t_ff in (0.001, 0.005, 0.02):
        m = DFIGFull(model.data, replace(Controls(), t_ff=t_ff))
        md, j, _, _ = _sso(m, 0.5, 1.0)
        sig.append(md.sigma[j])
    assert sig[0] > sig[1] > sig[2]


def test_only_the_network_mode_is_unstable(model):
    md, j, _, _ = _sso(model, 0.5, 1.0)
    conj = int(np.argmin(np.abs(md.eigenvalues - np.conj(md.eigenvalues[j]))))
    rest = np.delete(md.sigma, [j, conj])
    assert md.sigma[j] > 0 > rest.max()
    assert md.participation[[0, 1, 4, 5], j].sum() > 0.8  # line flux and capacitor voltage


# ------------------------------------------------------------------ time domain
def test_compensation_step_grows_at_the_eigenvalue_rate(model):
    d = model.data
    x_a, p_a = model.equilibrium(0.30 * d.x_e, w_r=1.0)
    p_b = {**p_a, "x_c": torch.as_tensor(0.35 * d.x_e, dtype=torch.float64)}
    _, traj = rk4(model.rhs, x_a, p_b, 0.6, 1e-4, save_every=5)
    md, j, x_b, _ = _sso(model, 0.35, 1.0)
    wave = model.outputs(traj, p_b)["p_t"].numpy()[200:] - float(model.outputs(x_b, p_b)["p_t"])
    pole = nearest_pole(matrix_pencil(wave, 5e-4, 10), abs(float(md.freq_hz[j])))
    assert abs(md.sigma[j] - 3.474) < 2e-3
    assert abs(pole.real - md.sigma[j]) < 0.05
    assert abs(pole.imag / (2 * np.pi) - abs(md.freq_hz[j])) < 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_batched_rk4_on_gpu_matches_cpu(model):
    cases = [model.equilibrium(k * model.data.x_e, w_r=w) for k, w in ((0.2, 0.8), (0.4, 1.0))]
    out = {}
    for dev in ("cpu", "cuda"):
        xs = torch.stack([c[0] for c in cases]).to(dev)
        ps = {key: torch.stack([c[1][key] for c in cases]).to(dev) for key in cases[0][1]}
        xs[:, 4] += 1e-3
        _, tr = rk4(model.rhs, xs, ps, 0.02, 1e-4, save_every=200)
        out[dev] = tr[-1].cpu()
    assert (out["cpu"] - out["cuda"]).abs().max() < 1e-11
