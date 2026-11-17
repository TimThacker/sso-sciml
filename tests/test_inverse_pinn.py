"""The inverse PINN's parameters (step 47). CPU."""

import numpy as np
import torch

from sso_sciml import inverse_pinn as ip
from sso_sciml.data import t2


def test_unscale_matches_the_numpy_version():
    theta = t2.sample(4)
    u = t2.scale_theta(theta)
    for row, th in zip(u, theta, strict=True):
        got = ip.unscale_torch(torch.as_tensor(row, dtype=torch.float64)).numpy()
        assert np.allclose(got, th)


def test_parameters_written_in_torch_match_those_of_the_plant():
    theta = np.array([0.3, 1.05, 12.0, 15.0])
    s = t2.setup(theta)
    th = torch.as_tensor(theta, dtype=torch.float64)
    base = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in s["p"].items()}
    q = ip.parameters(th, base["p_m"], base)
    for key in ("x_c", "kp", "ki", "pll_kp", "pll_ki", "out_kp", "out_ki", "p_m"):
        assert float(q[key]) == np.float64(s["p"][key]).item() or np.isclose(
            float(q[key]), s["p"][key], rtol=1e-12)
    model, _, _ = t2.plant(theta)
    x = torch.as_tensor(s["x_eq"], dtype=torch.float64)
    assert float(model.rhs(x, q).abs().max()) < 1e-8


def test_parameters_are_differentiable_in_u():
    u = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    base = {k: torch.as_tensor(v, dtype=torch.float64)
            for k, v in t2.setup(np.array([0.3, 1.0, 10.0, 10.0]))["p"].items()}
    q = ip.parameters(ip.unscale_torch(u), base["p_m"], base)
    (q["x_c"] + q["kp"] + q["pll_ki"]).backward()
    assert (u.grad[[0, 2, 3]] != 0).all() and u.grad[1] == 0
