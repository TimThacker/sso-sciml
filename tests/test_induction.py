"""Phase 2 checks on the induction generator model (System B1).

Each test pins a number from docs/steps/02-induction-generator.md.
"""

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from sso_sciml.physics import energy, slip
from sso_sciml.physics.induction import N_ELECTRICAL, W_R, InductionGenerator
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes

P_GEN = 0.9
MACHINE_KEYS = ("r_s", "x_ls", "x_m", "r_r", "x_lr")


@pytest.fixture(scope="module")
def model() -> InductionGenerator:
    return InductionGenerator()


def _hand(model, k, w_r, **ov):
    d = model.data
    q = {key: ov.get(key, getattr(d, key)) for key in (*MACHINE_KEYS, "r_e", "x_e")}
    return slip.analyse_induction(
        q["r_s"], q["x_ls"], q["x_m"], q["r_r"], q["x_lr"], q["r_e"], q["x_e"], k * q["x_e"],
        d.f0, w_r,
    )  # fmt: skip


def _full(model, k, **ov):
    x0, p = model.equilibrium(k * model.data.x_e, p_gen=P_GEN, **ov)
    return jacobian(model.rhs, x0, p).numpy(), x0, p


def _network_mode(model, lam, sr):
    target = complex(sr.sigma_circuit, 2 * np.pi * (model.data.f0 - sr.f_circuit))
    return complex(lam[np.argmin(np.abs(lam - target))])


def _sigma_full(model, k, **ov):
    a, x0, _ = _full(model, k, **ov)
    return _network_mode(model, np.linalg.eigvals(a), _hand(model, k, float(x0[W_R]), **ov)).real


# ------------------------------------------------------------------ steady state
@pytest.mark.parametrize("k", [0.3, 0.6, 0.95])
def test_equilibrium_is_a_fixed_point_and_delivers_the_power(model, k):
    x0, p = model.equilibrium(k * model.data.x_e, p_gen=P_GEN)
    assert model.rhs(x0, p).abs().max() < 1e-11
    c = model.currents(x0, p)
    assert abs(float(p["v_b"] * c["i_D"]) - P_GEN) < 1e-10  # bus voltage is on the D axis
    assert 1.0 < float(x0[W_R]) < 1.02  # generating: just above synchronous speed
    assert float(model.torque(x0, p)) > P_GEN  # shaft supplies the losses too


def test_power_beyond_pull_out_is_refused(model):
    with pytest.raises(ValueError, match="pull-out"):
        model.equilibrium(0.2 * model.data.x_e, p_gen=P_GEN)


def test_steady_state_power_balance(model):
    """Shaft power = bus power + copper losses in stator, rotor and line."""
    x0, p = model.equilibrium(0.6 * model.data.x_e, p_gen=P_GEN)
    c = model.currents(x0, p)
    i2 = float(c["i_D"] ** 2 + c["i_Q"] ** 2)
    ir2 = float(c["ir_D"] ** 2 + c["ir_Q"] ** 2)
    d = model.data
    shaft = float(model.torque(x0, p) * x0[W_R])
    assert abs(shaft - (P_GEN + (d.r_s + d.r_e) * i2 + d.r_r * ir2)) < 1e-12


def test_autodiff_jacobian_matches_finite_difference(model):
    a, x0, p = _full(model, 0.8)
    h = 1e-6
    fd = np.zeros_like(a)
    for j in range(x0.numel()):
        e = torch.zeros_like(x0)
        e[j] = h
        fd[:, j] = ((model.rhs(x0 + e, p) - model.rhs(x0 - e, p)) / (2 * h)).numpy()
    assert np.max(np.abs(a - fd)) < 1e-5


def test_rhs_is_differentiable_in_machine_parameters(model):
    x0, p = model.equilibrium(0.8 * model.data.x_e, p_gen=P_GEN)
    r_r = p["r_r"].clone().requires_grad_(True)
    out = model.rhs(x0 + 1e-3, {**p, "r_r": r_r})
    (g,) = torch.autograd.grad(out.pow(2).sum(), r_r)
    assert torch.isfinite(g) and g.abs() > 0


# ------------------------------------------------------------------ slip mechanism
@pytest.mark.parametrize("k", [0.3, 0.6, 0.9])
def test_circuit_root_equals_locked_speed_eigenvalue(model, k):
    """For a round rotor the equivalent circuit is exact once the speed is held."""
    a, x0, _ = _full(model, k)
    sr = _hand(model, k, float(x0[W_R]))
    lam = _network_mode(model, np.linalg.eigvals(a[:N_ELECTRICAL, :N_ELECTRICAL]), sr)
    assert abs(lam.real - sr.sigma_circuit) < 1e-9
    assert abs(lam.imag / (2 * np.pi) - (model.data.f0 - sr.f_circuit)) < 1e-9


def test_hand_growth_rate_tracks_the_full_model(model):
    for k in (0.3, 0.5, 0.7, 0.9):
        a, x0, _ = _full(model, k)
        sr = _hand(model, k, float(x0[W_R]))
        assert sr.slip < 0
        full = _network_mode(model, np.linalg.eigvals(a), sr).real
        assert abs(sr.sigma_hand - full) < 0.08


def test_base_machine_values_at_80_percent(model):
    a, x0, _ = _full(model, 0.8)
    sr = _hand(model, 0.8, float(x0[W_R]))
    assert abs(sr.f_er - 44.06) < 0.01
    assert abs(sr.slip - (-0.3695)) < 1e-3
    assert abs(sr.r_machine - (-0.00926)) < 1e-5
    assert abs(_network_mode(model, np.linalg.eigvals(a), sr).real - (-2.4388)) < 1e-3


def test_self_excitation_boundary_of_the_base_machine(model):
    """Hand calculation 97.4 %, solver 96.5 % compensation."""
    k_hand = brentq(
        lambda k: _hand(model, k, float(_full(model, k)[1][W_R])).r_net, 0.9, 0.99, xtol=1e-9
    )
    k_full = brentq(lambda k: _sigma_full(model, k), 0.9, 0.99, xtol=1e-9)
    assert abs(k_hand - 0.9739) < 5e-4
    assert abs(k_full - 0.9648) < 5e-4


def test_boundary_falls_as_rotor_resistance_rises(model):
    ks = [brentq(lambda k, r=r: _sigma_full(model, k, r_r=r), 0.3, 0.99) for r in (0.01, 0.02)]
    assert abs(ks[0] - 0.7298) < 1e-3
    assert abs(ks[1] - 0.4428) < 1e-3


def test_boundary_falls_as_rotor_speed_falls(model):
    """With the speed held, net resistance changes sign exactly where the eigenvalue does."""
    d = model.data
    for w_r, expected in ((0.7, 0.4724), (0.9, 0.7806)):

        def sigma(k, w_r=w_r):
            x = torch.zeros(7, dtype=torch.float64)
            x[W_R] = w_r
            a = jacobian(model.rhs, x, model.params(k * d.x_e)).numpy()
            return np.linalg.eigvals(a[:N_ELECTRICAL, :N_ELECTRICAL]).real.max()

        k_eig = brentq(sigma, expected - 0.05, expected + 0.05, xtol=1e-11)
        k_hand = brentq(
            lambda k, w_r=w_r: _hand(model, k, w_r).r_net, expected - 0.05, expected + 0.05,
            xtol=1e-11,
        )  # fmt: skip
        assert abs(k_eig - expected) < 1e-3
        assert abs(k_eig - k_hand) < 1e-8


# ------------------------------------------------------------------ energy accounting
def _energy_case(model, k, **ov):
    a, x0, p = _full(model, k, **ov)
    md = modes(a)
    w_r = float(x0[W_R])
    sr = _hand(model, k, w_r, **ov)
    target = complex(sr.sigma_circuit, 2 * np.pi * (model.data.f0 - sr.f_circuit))
    j = int(np.argmin(np.abs(md.eigenvalues - target)))
    v = md.right[:, j].real
    v = v / np.abs(v).max()
    _, traj = rk4(model.rhs, x0 + 1e-6 * torch.as_tensor(v), p, 0.2, 1e-4)
    acc = energy.account(model, traj, x0, p, 1e-4)
    mach = {key: ov.get(key, getattr(model.data, key)) for key in MACHINE_KEYS}
    s_mode = complex(md.sigma[j], -abs(md.eigenvalues[j].imag)) / model.data.wb + 1j
    return acc, sr, slip.induction_impedance(s_mode, w_r=w_r, **mach).real


def test_waveform_energy_matches_the_impedance_when_speed_is_held(model):
    acc, _, r_exact = _energy_case(model, 0.95, r_r=0.012, h=1e9)
    assert r_exact < 0
    assert abs(acc.r_effective - r_exact) / abs(r_exact) < 1e-7


def test_machine_feeds_energy_and_growth_needs_more_than_the_network_loses(model):
    grow, sr_g, _ = _energy_case(model, 0.95, r_r=0.012)
    decay, sr_d, _ = _energy_case(model, 0.8)
    assert grow.e_machine[-1] > grow.e_network_loss[-1] > 0
    assert 0 < decay.e_machine[-1] < decay.e_network_loss[-1]
    # Free rotor: the waveform resistance is within 15 % of the slip calculation.
    for acc, sr in ((grow, sr_g), (decay, sr_d)):
        assert abs(acc.r_effective - sr.r_machine) / abs(sr.r_machine) < 0.15


# ------------------------------------------------------------------ batching
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_batched_rk4_over_machine_parameters_on_gpu_matches_cpu(model):
    cases = [
        model.equilibrium(k * model.data.x_e, p_gen=P_GEN, r_r=r)
        for k, r in ((0.5, 0.005), (0.8, 0.01), (0.95, 0.012))
    ]
    out = {}
    for dev in ("cpu", "cuda"):
        xs = torch.stack([c[0] for c in cases]).to(dev)
        ps = {key: torch.stack([c[1][key] for c in cases]).to(dev) for key in cases[0][1]}
        xs[:, W_R] += 1e-3
        _, tr = rk4(model.rhs, xs, ps, 0.02, 2e-4, save_every=100)
        out[dev] = tr[-1].cpu()
    assert (out["cpu"] - out["cuda"]).abs().max() < 1e-12
