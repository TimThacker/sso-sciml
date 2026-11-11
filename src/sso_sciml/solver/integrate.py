"""Time integration: batched fixed-step RK4 in PyTorch and adaptive SciPy references."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.integrate import solve_ivp


@torch.no_grad()
def rk4(
    rhs: Callable, x0: torch.Tensor, p: dict, t_end: float, dt: float, save_every: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classical Runge-Kutta. x0 may be batched; returns (t, x) with x of shape (n_t, ..., n)."""
    n_steps = int(round(t_end / dt))
    x = x0.clone()
    out = [x.clone()]
    for k in range(1, n_steps + 1):
        k1 = rhs(x, p)
        k2 = rhs(x + 0.5 * dt * k1, p)
        k3 = rhs(x + 0.5 * dt * k2, p)
        k4 = rhs(x + dt * k3, p)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if k % save_every == 0:
            out.append(x.clone())
    t = torch.arange(len(out), dtype=torch.float64) * (dt * save_every)
    return t, torch.stack(out)


def reference(
    rhs: Callable,
    x0: torch.Tensor,
    p: dict,
    t_eval: np.ndarray,
    method: str = "Radau",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> np.ndarray:
    """Adaptive integration of a single (unbatched) case with SciPy. Returns (n_t, n)."""

    def f(_t, y):
        return rhs(torch.as_tensor(y, dtype=torch.float64), p).numpy()

    def jac(_t, y):
        yy = torch.as_tensor(y, dtype=torch.float64)
        return torch.func.jacrev(lambda v: rhs(v, p))(yy).numpy()

    kwargs = {"jac": jac} if method in ("Radau", "BDF", "LSODA") else {}
    sol = solve_ivp(
        f,
        (float(t_eval[0]), float(t_eval[-1])),
        x0.numpy(),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **kwargs,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T
