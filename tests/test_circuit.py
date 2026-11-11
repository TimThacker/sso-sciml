"""Phase 2 checks on the impedance formulation and the network study.

Each test pins a number from docs/steps/05-network-and-impedance.md.
"""

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from sso_sciml.physics import circuit as ct
from sso_sciml.physics import slip
from sso_sciml.physics.dfig import DFIG, gains_from_bandwidth
from sso_sciml.solver.linearize import jacobian

R_PLANT, X_PLANT = 0.010, 0.15


@pytest.fixture(scope="module")
def model() -> DFIG:
    return DFIG()


@pytest.fixture(scope="module")
def mach(model) -> dict:
    return {key: getattr(model.data, key) for key in ("r_s", "x_ls", "x_m", "r_r", "x_lr")}


def _growth(model, mach, k, w_r, bw, r_mult=1.0, plant=False, par=None, online=1.0):
    d = model.data
    kp, ki = gains_from_bandwidth(d, bw)
    z = ct.branch(r_mult * d.r_e, d.x_e, k * d.x_e)
    if par is not None:
        z = ct.parallel(z, ct.branch(par * d.r_e, par * d.x_e))
    if plant:
        z = z + ct.branch(R_PLANT, X_PLANT)
    z_m = ct.dfig_impedance(w_r=w_r, kp=kp, ki=ki, **mach) / online
    return ct.sso_growth(ct.modes(z_m + z))


# ------------------------------------------------------------------ algebra
def test_rational_arithmetic():
    s = 0.3 + 0.7j
    a, b = ct.branch(0.02, 0.5, 0.2), ct.branch(0.01, 0.3)
    assert abs(a(s) - (0.02 + 0.5 * s + 0.2 / s)) < 1e-14
    assert abs(ct.parallel(a, b)(s) - a(s) * b(s) / (a(s) + b(s))) < 1e-13
    assert abs((a / 0.5 + 2.0)(s) - (a(s) / 0.5 + 2.0)) < 1e-13


def test_polynomial_machine_impedance_matches_the_direct_expression(mach):
    for s in (0.6j, 0.02 + 0.75j, -0.01 + 0.4j):
        z1 = ct.dfig_impedance(w_r=0.9, kp=0.03, ki=0.4, **mach)(s)
        z2 = slip.dfig_impedance(s, w_r=0.9, kp=0.03, ki=0.4, **mach)
        assert abs(z1 - z2) < 1e-12


def test_roots_drop_cancelled_factors():
    # (s - 1)(s - 2) / (s - 2) has the single zero s = 1.
    z = ct.poly(2.0, -3.0, 1.0) / ct.poly(-2.0, 1.0)
    r = ct.roots(z)
    assert r.size == 1 and abs(r[0] - 1.0) < 1e-9


# ------------------------------------------------------------------ against the state-space model
@pytest.mark.parametrize("w_r", [0.75, 1.0, 1.2])
def test_impedance_roots_equal_state_space_eigenvalues(model, mach, w_r):
    d = model.data
    for k, bw in ((0.2, 3.0), (0.5, 10.0), (0.8, 30.0)):
        kp, ki = gains_from_bandwidth(d, bw)
        z = ct.dfig_impedance(w_r=w_r, kp=kp, ki=ki, **mach) + ct.branch(d.r_e, d.x_e, k * d.x_e)
        lam = ct.modes(z)
        p = model.params(k * d.x_e, w_r=w_r, kp=kp, ki=ki)
        ss = np.linalg.eigvals(jacobian(model.rhs, torch.zeros(8, dtype=torch.float64), p).numpy())
        assert lam.size == 4
        assert max(np.abs(ss - v).min() for v in lam) < 1e-6


# ------------------------------------------------------------------ network study
def test_growth_rates_by_network(model, mach):
    got = [
        _growth(model, mach, 0.5, 1.0, 10.0),
        _growth(model, mach, 0.5, 1.0, 10.0, r_mult=2.0),
        _growth(model, mach, 0.5, 1.0, 10.0, plant=True),
        _growth(model, mach, 0.5, 1.0, 10.0, par=1.0),
        _growth(model, mach, 0.5, 1.0, 10.0, plant=True, par=1.0),
    ]
    assert np.allclose(got, [8.637, 3.423, 1.643, 1.524, -3.386], atol=2e-3)


def test_largest_stable_bandwidth(model, mach):
    radial = brentq(lambda f: _growth(model, mach, 0.5, 1.0, f), 1.0, 10.0, xtol=1e-9)
    meshed = brentq(
        lambda f: _growth(model, mach, 0.5, 1.0, f, plant=True, par=1.0), 5.0, 50.0, xtol=1e-9
    )
    assert abs(radial - 3.45) < 0.01
    assert abs(meshed - 19.19) < 0.02


def test_partial_plant_is_worse_than_full_plant(model, mach):
    g = {n: _growth(model, mach, 0.5, 1.0, 10.0, online=n) for n in (0.05, 0.45, 1.0)}
    assert g[0.45] > g[1.0] > g[0.05]
    assert abs(g[0.45] - 10.74) < 0.02
