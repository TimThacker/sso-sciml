"""Protection of the doubly-fed plant (step 33)."""

import numpy as np
import pytest
import torch

from sso_sciml import protection as prot
from sso_sciml.physics.dfig_full import DFIGFull


@pytest.fixture(scope="module")
def case():
    model = DFIGFull()
    x0, p = model.equilibrium(0.5 * model.data.x_e, w_r=1.0)
    return model, x0, p


def _t(v):
    return torch.tensor(v, dtype=torch.float64)


def test_protection_does_nothing_while_no_limit_acts(case):
    model, x0, p = case
    x = x0 + 1e-3
    on = {**p, **{k: _t(v) for k, v in prot.SETTINGS.items()}}
    assert torch.equal(model.rhs(x, p), model.rhs(x, on))
    assert float(model.rhs(x0, on).abs().max()) < 1e-9


def test_chopper_draws_its_power_from_the_link(case):
    model, x0, p = case
    x = x0.clone()
    x[prot.V_DC] = 1.2
    on = {**p, "chop_on": _t(1.10), "chop_g": _t(0.8)}
    assert float(model.outputs(x, on)["p_chop"]) == pytest.approx(0.8 * 1.2**2)
    diff = model.rhs(x, on) - model.rhs(x, p)
    assert float(diff[prot.V_DC]) == pytest.approx(-0.8 * 1.2**2 / (float(p["c_dc"]) * 1.2))
    assert float(diff.abs().sum()) == pytest.approx(abs(float(diff[prot.V_DC])))
    # Half way up the switching band it conducts half.
    x[prot.V_DC] = 1.125
    assert float(model.outputs(x, on)["p_chop"]) == pytest.approx(0.5 * 0.8 * 1.125**2)


def test_current_limit_holds_the_reference_and_stops_the_outer_integrators(case):
    model, x0, p = case
    x = x0 + 1e-3
    size = float(prot.signals(model, x, p)[prot.SIGNALS.index("i_reference")])
    tight = {**p, "ir_max": _t(0.5 * size), "aw": _t(1.0)}
    free, held = model.rhs(x, p), model.rhs(x, tight)
    # The outer integrators (15, 16) run at half speed, and the current controller sees
    # a reference of half the size.
    assert float(held[15] / free[15]) == pytest.approx(0.5, rel=1e-9)
    assert float(held[16] / free[16]) == pytest.approx(0.5, rel=1e-9)
    assert not torch.equal(free[11:13], held[11:13])


def test_anti_windup_pulls_the_current_integrators_toward_the_limited_command(case):
    model, x0, p = case
    size = float(torch.hypot(*model.outputs(x0, p)["vr_command"]))
    limit = {**p, "vr_max": _t(0.5 * size)}
    wound, held = model.rhs(x0, limit), model.rhs(x0, {**limit, "aw": _t(1.0)})
    cmd = torch.stack(model.outputs(x0, p)["vr_command"])
    expect = (0.5 - 1.0) * cmd / p["kp"]
    assert torch.allclose(held[11:13] - wound[11:13], expect, atol=1e-12)


def test_first_outside_waits_for_the_delay():
    t = np.arange(0.0, 1.0, 0.01)
    value = np.ones((t.size, 2))
    value[20:25, 0] = 2.0  # 0.05 s outside: too short for a delay of 0.1 s
    value[50:, 0] = 2.0
    value[30:, 1] = 0.0
    assert prot.first_outside(t, value, 0.5, 1.5, 0.0) == pytest.approx([0.20, 0.30])
    assert prot.first_outside(t, value, 0.5, 1.5, 0.1) == pytest.approx([0.60, 0.40])


def test_run_is_the_plain_integrator_while_nothing_fails(case):
    from sso_sciml.solver.integrate import rk4

    model, x0, p = case
    start = (x0 + 1e-3)[None]
    q = {k: v[None] for k, v in p.items()}
    t, sig, failed = prot.run(model, start, q, 0.02, 5e-4, every=2)
    _, ref = rk4(model.rhs, start, q, 0.02, 5e-4, save_every=2)
    assert t.size == ref.shape[0] and np.isnan(failed).all()
    assert np.allclose(sig[:, 0, 0], ref[:, 0, prot.V_DC].numpy(), atol=1e-14)
