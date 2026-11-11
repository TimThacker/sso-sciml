"""Phase 1 checks on the solver core and the First Benchmark Model.

Each test pins a number. If a sign, a base, or a parameter is wrong, one of these fails.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from sso_sciml.physics import slip, torsional
from sso_sciml.physics.fbm import FBM, GEN, N_ELECTRICAL, stack_cases
from sso_sciml.physics.shaft import torsional_modes
from sso_sciml.solver.integrate import reference, rk4
from sso_sciml.solver.linearize import jacobian, modes, newton_equilibrium
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

PUBLISHED_HZ = np.array([15.71, 20.21, 25.55, 32.28, 47.46])
DW_GEN = N_ELECTRICAL + 6 + GEN


@pytest.fixture(scope="module")
def model() -> FBM:
    return FBM()


@pytest.fixture(scope="module")
def shaft(model):
    d = model.data
    return torsional_modes(np.array(d.inertia_h), np.array(d.stiffness_k), d.f0)


def _matrix(model, x_c, **kw):
    x0, p = model.equilibrium(x_c, **kw)
    return jacobian(model.rhs, x0, p).numpy(), x0, p


def _locked_mode(a):
    lam = np.linalg.eigvals(a[:N_ELECTRICAL, :N_ELECTRICAL])
    cand = lam[(lam.imag > 2 * np.pi) & (lam.imag < 2 * np.pi * 59)]
    return cand[np.argmax(cand.real)]


def _torsional_sigma(a, f_n):
    md = modes(a)
    shaft_part = md.participation[N_ELECTRICAL:].sum(axis=0)
    cand = np.where((np.abs(md.freq_hz - f_n) < 1.5) & (md.freq_hz > 0))[0]
    return md.sigma[cand[np.argmax(shaft_part[cand])]]


# ------------------------------------------------------------------ shaft
def test_torsional_frequencies_match_published(shaft):
    freq, _ = shaft
    assert abs(freq[0]) < 1e-6  # rigid-body mode
    assert np.max(np.abs(freq[1:] - PUBLISHED_HZ)) < 0.01


def test_mode_five_barely_moves_the_generator(shaft):
    _, shapes = shaft
    assert abs(shapes[GEN, 5]) < 0.01


# ------------------------------------------------------------------ machine circuits
def test_operational_reactance_limits(model):
    d, q, dat = model.d, model.q, model.data
    assert abs(d.operational_reactance(1e-9j) - dat.xd) < 1e-5
    assert abs(d.operational_reactance(1e9j) - dat.xd_pp) < 1e-6
    assert abs(q.operational_reactance(1e-9j) - dat.xq) < 1e-5
    assert abs(q.operational_reactance(1e9j) - dat.xq_pp) < 1e-6


# ------------------------------------------------------------------ steady state
@pytest.mark.parametrize("x_c", [0.05, 0.371, 0.6])
def test_equilibrium_is_a_fixed_point(model, x_c):
    x0, p = model.equilibrium(x_c)
    assert model.rhs(x0, p).abs().max() < 1e-11
    assert abs(float(model.torque(x0, p)) - 0.9) < 1e-12  # Ra = 0, so torque equals power


def test_newton_recovers_equilibrium(model):
    x0, p = model.equilibrium(0.371)
    start = x0 + 1e-3 * torch.linspace(-1, 1, x0.numel(), dtype=torch.float64)
    assert (newton_equilibrium(model.rhs, start, p) - x0).abs().max() < 1e-9


def test_autodiff_jacobian_matches_finite_difference(model):
    a, x0, p = _matrix(model, 0.371)
    h = 1e-6
    fd = np.zeros_like(a)
    for j in range(x0.numel()):
        e = torch.zeros_like(x0)
        e[j] = h
        fd[:, j] = ((model.rhs(x0 + e, p) - model.rhs(x0 - e, p)) / (2 * h)).numpy()
    assert np.max(np.abs(a - fd)) < 1e-5


# ------------------------------------------------------------------ network alone
def test_series_rlc_modes_appear_at_f0_minus_and_plus_fer(model):
    """With the rotor fluxes held constant, the loop is a series R-L-C.

    Frozen rotor flux makes the machine a constant subtransient reactance. Setting
    Xq'' = Xd'' removes saliency, so the result is exact: in the rotating frame the two
    modes sit at f0 - f_er and f0 + f_er and both decay at R * wb / (2 X).
    """
    dat = model.data
    x_total = dat.x_e + dat.xd_pp
    m = FBM(replace(dat, xq_pp=dat.xd_pp))
    x_c = 0.4
    a, _, _ = _matrix(m, x_c)
    lam = np.linalg.eigvals(a[[0, 1, 6, 7]][:, [0, 1, 6, 7]])
    f_er = dat.f0 * np.sqrt(x_c / x_total)
    got = np.sort(np.abs(lam.imag) / (2 * np.pi))
    assert np.allclose(got, np.sort([dat.f0 - f_er] * 2 + [dat.f0 + f_er] * 2), atol=0.02)
    assert np.allclose(lam.real, -dat.r_e * dat.wb / (2 * x_total), rtol=2e-3)


# ------------------------------------------------------------------ slip mechanism
@pytest.mark.parametrize("x_c", [0.1, 0.2, 0.371, 0.5])
def test_slip_hand_estimate_matches_locked_shaft_eigenvalue(model, x_c):
    dat = model.data
    a, _, _ = _matrix(model, x_c)
    lam = _locked_mode(a)
    sr = slip.analyse(model.d, model.q, dat.r_e, dat.x_e, x_c, dat.f0)
    assert sr.slip < 0 and sr.r_machine < 0
    assert abs((dat.f0 - sr.f_er) - lam.imag / (2 * np.pi)) < 0.1
    assert abs(sr.sigma_hand - lam.real) / abs(lam.real) < 0.03


def test_benchmark_is_not_self_excited_by_slip_alone(model):
    dat = model.data
    for x_c in np.linspace(0.05, 0.6, 12):
        assert not slip.analyse(model.d, model.q, dat.r_e, dat.x_e, float(x_c), dat.f0).self_excited


def test_self_excitation_threshold(model):
    """Below the resistance where R_net = 0, the locked-shaft network mode must grow."""
    dat = model.data
    r_star = -slip.analyse(model.d, model.q, 0.0, dat.x_e, 0.371, dat.f0).r_machine
    assert abs(r_star - 0.00613) < 2e-5
    below, _, _ = _matrix(model, 0.371, r_e=0.9 * r_star)
    above, _, _ = _matrix(model, 0.371, r_e=1.1 * r_star)
    assert _locked_mode(below).real > 0 > _locked_mode(above).real


# ------------------------------------------------------------------ torsional interaction
def test_benchmark_compensation_destabilises_mode_two(model, shaft):
    freq, _ = shaft
    a, _, _ = _matrix(model, 0.371)
    assert abs(_torsional_sigma(a, freq[2]) - 0.3578) < 2e-3
    assert abs(_torsional_sigma(a, freq[5])) < 1e-4  # mode 5 is not excited


def test_damping_torque_estimate_tracks_eigenvalue_when_coupling_is_weak(model, shaft):
    freq, shapes = shaft
    dat = model.data
    h = np.array(dat.inertia_h)
    for x_c, n in [(0.10, 4), (0.25, 3), (0.33, 2), (0.30, 1)]:
        a, _, _ = _matrix(model, x_c)
        eig = _torsional_sigma(a, freq[n])
        est = torsional.predicted_growth(a, h, shapes[:, n], freq[n], dat.wb)
        assert np.sign(eig) == np.sign(est)
        assert abs(eig - est) < 0.15 * abs(eig) + 1e-3


# ------------------------------------------------------------------ integration
def test_matrix_pencil_recovers_known_poles():
    t = np.arange(0, 1.0, 1e-3)
    y = np.exp(-0.7 * t) * np.cos(2 * np.pi * 20.2 * t + 0.3) + 0.5 * np.exp(0.4 * t) * np.sin(
        2 * np.pi * 31.0 * t
    )
    poles = matrix_pencil(y, 1e-3, 4)
    p1, p2 = nearest_pole(poles, 20.2), nearest_pole(poles, 31.0)
    assert abs(p1 - complex(-0.7, 2 * np.pi * 20.2)) < 1e-6
    assert abs(p2 - complex(0.4, 2 * np.pi * 31.0)) < 1e-6


def test_time_domain_growth_matches_eigenvalue(model, shaft):
    freq, _ = shaft
    a, x0, p = _matrix(model, 0.371)
    md = modes(a)
    shaft_part = md.participation[N_ELECTRICAL:].sum(axis=0)
    cand = np.where((np.abs(md.freq_hz - freq[2]) < 1.5) & (md.freq_hz > 0))[0]
    k = cand[np.argmax(shaft_part[cand])]
    v = md.right[:, k].real
    v = v / np.abs(v).max()
    probe = int(np.argmax(np.abs(v)))
    eps = 1e-6
    _, traj = rk4(model.rhs, x0 + eps * torch.as_tensor(v), p, 0.5, 2e-4, save_every=5)
    y = (traj[:, probe] - x0[probe]).numpy() / eps
    pole = nearest_pole(matrix_pencil(y, 1e-3, 4), md.freq_hz[k])
    assert abs(pole.real - md.sigma[k]) < 1e-4
    assert abs(pole.imag / (2 * np.pi) - md.freq_hz[k]) < 1e-4


def test_rk4_agrees_with_adaptive_reference(model):
    _, x0, p = _matrix(model, 0.371)
    xk = x0.clone()
    xk[DW_GEN] += 0.01
    t_eval = np.linspace(0, 0.1, 101)
    ref = reference(model.rhs, xk, p, t_eval, method="DOP853", rtol=1e-11, atol=1e-13)
    _, tr = rk4(model.rhs, xk, p, 0.1, 1e-4, save_every=10)
    assert np.abs(tr.numpy() - ref).max() < 1e-7


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_batched_rk4_on_gpu_matches_cpu(model):
    cases = [model.equilibrium(xc) for xc in (0.1, 0.25, 0.371, 0.5)]
    out = {}
    for dev in ("cpu", "cuda"):
        xs, ps = stack_cases(cases, device=dev)
        xs[:, DW_GEN] += 0.01
        _, tr = rk4(model.rhs, xs, ps, 0.02, 2e-4, save_every=100)
        out[dev] = tr[-1].cpu()
    assert (out["cpu"] - out["cuda"]).abs().max() < 1e-12
