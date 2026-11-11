"""Phase 2 checks on the full-control model with plant impedance and a parallel path.

Each test pins a number from docs/steps/05-network-and-impedance.md.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from sso_sciml.physics import circuit as ct
from sso_sciml.physics.dfig import gains_from_bandwidth
from sso_sciml.physics.dfig_full import Controls, DFIGFull
from sso_sciml.physics.dfig_meshed import N_STATE, STATE_NAMES, DFIGMeshed
from sso_sciml.solver.integrate import rk4
from sso_sciml.solver.linearize import jacobian, modes
from sso_sciml.solver.modal import matrix_pencil, nearest_pole

OFF = {"pll_hz": 1e-3, "outer_hz": 1e-3, "x_f": 1e3, "dc_hz": 1e-3}


def _sso(model, k, w_r, **over):
    x0, p = model.equilibrium(k * model.data.x_e, w_r=w_r, **over)
    md = modes(jacobian(model.rhs, x0, p))
    cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 58.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    return md, j, x0, p


@pytest.fixture(scope="module")
def reference() -> DFIGMeshed:
    return DFIGMeshed()  # plant impedance plus a parallel path equal to the line


def test_state_vector_layout():
    assert len(STATE_NAMES) == N_STATE == 25


@pytest.mark.parametrize("w_r", [0.75, 1.0, 1.2])
def test_equilibrium_is_a_fixed_point_and_currents_add_up(reference, w_r):
    x0, p = reference.equilibrium(0.5 * reference.data.x_e, w_r=w_r)
    assert reference.rhs(x0, p).abs().max() < 1e-10
    c = reference.currents(x0, p)
    assert abs(float(c["i_D"] + c["ig_D"] - c["ic_D"] - c["ip_D"])) < 1e-13
    out = reference.outputs(x0, p)
    assert abs(float(out["p_t"]) - (w_r / 1.2) ** 3) < 1e-10


def test_weak_parallel_path_and_no_plant_impedance_is_the_radial_model():
    full = DFIGFull()
    radial = DFIGMeshed(r_t=0.0, x_t=0.0, par_ratio=1e6)
    for k, w_r in ((0.3, 0.75), (0.7, 1.2)):
        a, ja, _, _ = _sso(full, k, w_r)
        b, jb, _, _ = _sso(radial, k, w_r)
        assert abs(a.sigma[ja] - b.sigma[jb]) < 1e-4


def test_controls_off_matches_the_impedance_model_on_the_meshed_network():
    d = DFIGFull().data
    mach = {key: getattr(d, key) for key in ("r_s", "x_ls", "x_m", "r_r", "x_lr")}
    kp, ki = gains_from_bandwidth(d, 10.0)
    m = DFIGMeshed(controls=replace(Controls(), **OFF))
    for w_r in (0.75, 1.0, 1.2):
        z = ct.parallel(ct.branch(d.r_e, d.x_e, 0.5 * d.x_e), ct.branch(d.r_e, d.x_e))
        z = z + ct.branch(m.r_t, m.x_t)
        imp = ct.sso_growth(ct.modes(ct.dfig_impedance(w_r=w_r, kp=kp, ki=ki, **mach) + z))
        md, j, _, _ = _sso(m, 0.5, w_r, h=1e6)
        assert abs(md.sigma[j] - imp) < 2e-3


def test_growth_rates_on_the_reference_network(reference):
    got = [float(_sso(reference, 0.5, w)[0].sigma[_sso(reference, 0.5, w)[1]])
           for w in (0.75, 1.0, 1.2)]  # fmt: skip
    assert np.allclose(got, [0.110, -3.460, -4.995], atol=3e-3)


def test_compensation_step_on_the_reference_network(reference):
    d = reference.data
    x_a, p_a = reference.equilibrium(0.45 * d.x_e, w_r=0.75)
    p_b = {**p_a, "x_c": torch.as_tensor(0.60 * d.x_e, dtype=torch.float64)}
    _, traj = rk4(reference.rhs, x_a, p_b, 0.8, 1e-4, save_every=5)
    md, j, x_b, _ = _sso(reference, 0.60, 0.75)
    wave = reference.outputs(traj, p_b)["p_t"].numpy()[300:]
    wave = wave - float(reference.outputs(x_b, p_b)["p_t"])
    pole = nearest_pole(matrix_pencil(wave, 5e-4, 10), abs(float(md.freq_hz[j])))
    assert abs(md.sigma[j] - 1.616) < 2e-3
    assert abs(pole.real - md.sigma[j]) < 0.02
