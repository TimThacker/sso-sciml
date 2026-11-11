"""Phase 2 checks on the doubly-fed machine with rotor current control (System B2, stage 1).

Each test pins a number from docs/steps/03-dfig-current-control.md.
"""

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from sso_sciml.physics import energy, slip
from sso_sciml.physics.dfig import DFIG, gains_from_bandwidth
from sso_sciml.physics.induction import N_ELECTRICAL, W_R, InductionGenerator
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes


@pytest.fixture(scope="module")
def model() -> DFIG:
    return DFIG()


@pytest.fixture(scope="module")
def mach(model) -> dict:
    return {key: getattr(model.data, key) for key in ("r_s", "x_ls", "x_m", "r_r", "x_lr")}


def _eig(model, k, w_r, kp, ki):
    p = model.params(k * model.data.x_e, w_r=w_r, kp=kp, ki=ki)
    return np.linalg.eigvals(jacobian(model.rhs, torch.zeros(8, dtype=torch.float64), p).numpy())


def _sso_sigma(lam):
    f = np.abs(lam.imag) / (2 * np.pi)
    return float(lam.real[(f > 2.0) & (f < 58.0)].max())


def _hand(model, mach, k, w_r, kp):
    d = model.data
    return slip.analyse_loop(
        lambda s: slip.dfig_impedance(s, w_r=w_r, kp=kp, ki=0.0, **mach),
        d.r_e, d.x_e, k * d.x_e, d.f0, w_r,
    )  # fmt: skip


def _tune(model):
    kp, ki = gains_from_bandwidth(model.data, 10.0)
    return kp, ki, ki / kp


# ------------------------------------------------------------------ model
def test_gains_from_bandwidth(model):
    kp, ki = gains_from_bandwidth(model.data, 10.0)
    assert abs(kp - 0.03164) < 1e-5
    assert abs(ki - 0.34495) < 1e-5


@pytest.mark.parametrize("ki", [0.0, 0.345])
def test_equilibrium_is_a_fixed_point_and_delivers_the_power(model, ki):
    x0, p = model.equilibrium(0.5 * model.data.x_e, w_r=0.8, kp=0.03, ki=ki, p_gen=0.9, q_gen=0.1)
    assert model.rhs(x0, p).abs().max() < 1e-12
    c = model.currents(x0, p)
    assert abs(float(c["i_D"]) - 0.9) < 1e-12 and abs(float(c["i_Q"]) + 0.1) < 1e-12


@pytest.mark.parametrize("w_r", [0.7, 1.0, 1.2])
def test_zero_gain_reproduces_the_induction_machine(model, w_r):
    ig = InductionGenerator()
    k = 0.6
    x = torch.zeros(7, dtype=torch.float64)
    x[W_R] = w_r
    a = jacobian(ig.rhs, x, ig.params(k * ig.data.x_e)).numpy()[:N_ELECTRICAL, :N_ELECTRICAL]
    lam = _eig(model, k, w_r, 0.0, 0.0)
    for mu in np.linalg.eigvals(a):
        assert np.abs(lam - mu).min() < 1e-9


def test_rhs_is_differentiable_in_the_controller_gain(model):
    x0, p = model.equilibrium(0.5 * model.data.x_e, w_r=1.0, kp=0.03, ki=0.3)
    kp = p["kp"].clone().requires_grad_(True)
    out = model.rhs(x0 + 1e-3, {**p, "kp": kp})
    (g,) = torch.autograd.grad(out.pow(2).sum(), kp)
    assert torch.isfinite(g) and g.abs() > 0


# ------------------------------------------------------------------ mechanism
def test_every_eigenvalue_is_a_root_of_the_loop_impedance(model, mach):
    """Kp + Ki/S in series with the rotor resistance is the exact circuit of this model."""
    d = model.data
    for k, w_r, kp, ki in ((0.2, 0.75, 0.01, 0.1), (0.5, 1.0, 0.05, 3.0), (0.8, 1.2, 0.3, 100.0)):
        for lam in _eig(model, k, w_r, kp, ki):
            z = min(
                abs(
                    d.r_e
                    + s * d.x_e
                    + k * d.x_e / s
                    + slip.dfig_impedance(s, w_r=w_r, kp=kp, ki=ki, **mach)
                )
                for s in (lam / d.wb + 1j, np.conj(lam) / d.wb + 1j)
            )
            assert z < 1e-8


def test_gain_adds_negative_resistance(model, mach):
    r = [_hand(model, mach, 0.5, 0.75, kp).r_machine for kp in (0.0, 0.05, 0.1)]
    assert r[0] > r[1] > r[2]
    assert abs(r[0] - (-0.0130)) < 2e-4
    assert abs(r[2] - (-0.269)) < 2e-3


def test_growth_rate_at_reference_tuning_falls_with_rotor_speed(model):
    kp, ki, _ = _tune(model)
    sig = [_sso_sigma(_eig(model, 0.5, w, kp, ki)) for w in (0.75, 1.0, 1.2)]
    assert np.allclose(sig, [20.543, 8.637, 3.752], atol=2e-3)


def test_growth_rate_rises_with_compensation_and_gain(model):
    kp, ki, ratio = _tune(model)
    by_k = [_sso_sigma(_eig(model, k, 1.0, kp, ki)) for k in (0.2, 0.4, 0.6, 0.8)]
    by_kp = [_sso_sigma(_eig(model, 0.5, 1.0, g, ratio * g)) for g in (0.01, 0.03, 0.1, 0.3)]
    assert np.all(np.diff(by_k) > 0) and np.all(np.diff(by_kp) > 0)


# ------------------------------------------------------------------ boundary
def _critical_gains(model, mach, k, w_r, ratio):
    hand = brentq(lambda g: _hand(model, mach, k, w_r, g).r_net, 1e-5, 0.3, xtol=1e-12)
    solver = brentq(lambda g: _sso_sigma(_eig(model, k, w_r, g, ratio * g)), 1e-5, 0.3, xtol=1e-12)
    return hand, solver


def test_largest_stable_gain_hand_against_solver(model, mach):
    _, _, ratio = _tune(model)
    hand, solver = _critical_gains(model, mach, 0.4, 1.0, ratio)
    assert abs(hand - 0.0156) < 2e-4
    assert abs(solver - 0.0155) < 2e-4
    assert abs(hand - solver) / solver < 0.02


def test_resistance_picture_fails_when_integral_action_is_strong(model, mach):
    _, _, ratio = _tune(model)
    hand, solver = _critical_gains(model, mach, 0.4, 1.0, 100 * ratio)
    assert (hand - solver) / solver > 0.5  # the hand calculation is optimistic


# ------------------------------------------------------------------ time domain and energy
def test_energy_budget_closes_and_the_converter_absorbs(model, mach):
    kp, ki, _ = _tune(model)
    k, w_r = 0.3, 1.0
    x0, p = model.equilibrium(k * model.data.x_e, w_r=w_r, kp=kp, ki=ki)
    md = modes(jacobian(model.rhs, x0, p))
    f = np.abs(md.freq_hz)
    cand = np.where((f > 2.0) & (f < 58.0) & (md.freq_hz > 0))[0]
    j = int(cand[np.argmax(md.sigma[cand])])
    assert abs(md.sigma[j] - 2.3485) < 1e-3
    v = md.right[:, j].real
    v = v / np.abs(v).max()
    _, traj = rk4(model.rhs, x0 + 1e-6 * torch.as_tensor(v), p, 0.3, 1e-4)

    acc = energy.account(model, traj, x0, p, 1e-4)
    s_mode = complex(md.sigma[j], -abs(md.eigenvalues[j].imag)) / model.data.wb + 1j
    r_exact = slip.dfig_impedance(s_mode, w_r=w_r, kp=kp, ki=ki, **mach).real
    assert abs(acc.r_effective - r_exact) / abs(r_exact) < 1e-7

    fl = energy.flows(model, traj, x0, p, 1e-4)
    assert abs(fl.closure) < 1e-5
    assert fl.to_network > 0
    assert fl.from_converter < 0  # the converter takes energy out, like a resistor
    assert fl.from_shaft > fl.to_network  # the shaft supplies all of it


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_batched_rk4_over_gains_on_gpu_matches_cpu(model):
    cases = [
        model.equilibrium(0.5 * model.data.x_e, w_r=w, kp=g, ki=11 * g)
        for w, g in ((0.75, 0.005), (1.0, 0.03), (1.2, 0.1))
    ]
    out = {}
    for dev in ("cpu", "cuda"):
        xs = torch.stack([c[0] for c in cases]).to(dev)
        ps = {key: torch.stack([c[1][key] for c in cases]).to(dev) for key in cases[0][1]}
        xs[:, 0] += 1e-3
        _, tr = rk4(model.rhs, xs, ps, 0.02, 2e-4, save_every=100)
        out[dev] = tr[-1].cpu()
    assert (out["cpu"] - out["cuda"]).abs().max() < 1e-12
