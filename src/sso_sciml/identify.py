"""Plant parameters from a recorded phase current, through the simulator (task T3).

The four parameters of dataset T2 are handled in the scaled units of `t2.scale_theta`,
in which the training range of each is [-1, 1]. Derivatives of the waveform with
respect to them are central differences of runs of the full model.

    jacobian       d waveform / d parameter for each case
    bound          the Cramer-Rao bound of each parameter for white noise of a given size
    gauss_newton   damped Gauss-Newton fit of the parameters to a recorded waveform
"""

from __future__ import annotations

import numpy as np

from . import measure
from .data import t2


def unscale(u: np.ndarray) -> np.ndarray:
    """Inverse of `t2.scale_theta`."""
    out = np.empty_like(u, dtype=float)
    for j, par in enumerate(t2.SPEC):
        lo, hi = (np.log(par.low), np.log(par.high)) if par.log else (par.low, par.high)
        v = 0.5 * (u[:, j] * (hi - lo) + lo + hi)
        out[:, j] = np.exp(v) if par.log else v
    return out


def waveform(u: np.ndarray, device: str = "cuda", processes: int = 8):
    """The deviation of the phase current from the fundamental for each row of scaled
    parameters, (cases, samples), and whether the case ran through the whole window."""
    d = t2.generate(unscale(u), processes=processes, device=device)
    ok = d["solved"].astype(bool) & (d["valid"] >= t2.N_T)
    obs = np.where(ok[:, None, None], d["observed"], 0.0)
    zero = measure.phase_current(np.zeros((1, t2.N_T, 5)))
    return measure.phase_current(obs) - zero, ok


def jacobian(u: np.ndarray, h: float = 1e-3, device: str = "cuda", processes: int = 8):
    """Central differences: (cases, samples, 4), and whether all eight runs of a case
    went through."""
    n, k = u.shape
    step = np.concatenate([s * h * np.eye(k) for s in (1.0, -1.0)])
    y, ok = waveform((u[:, None, :] + step[None, :, :]).reshape(-1, k), device, processes)
    y, ok = y.reshape(n, 2 * k, -1), ok.reshape(n, 2 * k).all(axis=1)
    return ((y[:, :k] - y[:, k:]) / (2 * h)).transpose(0, 2, 1), ok


def bound(jac: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Cramer-Rao bound on the standard deviation of each scaled parameter, (cases, 4),
    for white noise of standard deviation `noise` (one per case) on every sample."""
    fisher = np.einsum("nti,ntj->nij", jac, jac) / noise[:, None, None] ** 2
    return np.sqrt(np.diagonal(np.linalg.pinv(fisher, hermitian=True), axis1=1, axis2=2))


def gauss_newton(y: np.ndarray, u0: np.ndarray, iterations: int = 6, damping: float = 1e-3,
                 device: str = "cuda", processes: int = 8):  # fmt: skip
    """Fit the scaled parameters of each case to its recorded deviation `y`, starting
    from `u0`. A step is kept only where it lowers the misfit. Returns the parameters
    and the root-mean-square misfit of each case after every iteration."""
    u = u0.copy()
    w, ok = waveform(u, device, processes)
    cost = np.where(ok, np.sqrt(((y - w) ** 2).mean(axis=1)), np.inf)
    lam = np.full(len(u), damping)
    history = [cost.copy()]
    for _ in range(iterations):
        jac, good = jacobian(u, device=device, processes=processes)
        jtj = np.einsum("nti,ntj->nij", jac, jac)
        rhs = np.einsum("nti,nt->ni", jac, y - w)
        scale = np.einsum("nii->ni", jtj)
        lhs = jtj + (lam[:, None] * scale)[:, :, None] * np.eye(u.shape[1])
        step = np.zeros_like(u)
        use = good & ok & np.isfinite(lhs).all(axis=(1, 2))
        step[use] = np.linalg.solve(lhs[use], rhs[use][:, :, None])[:, :, 0]
        trial = np.clip(u + step, -1.6, 1.6)
        w_new, ok_new = waveform(trial, device, processes)
        cost_new = np.where(ok_new, np.sqrt(((y - w_new) ** 2).mean(axis=1)), np.inf)
        better = cost_new < cost
        u[better], w[better], cost[better] = trial[better], w_new[better], cost_new[better]
        ok = ok | better
        lam = np.where(better, lam / 3.0, lam * 10.0)
        history.append(cost.copy())
    return u, np.array(history)
