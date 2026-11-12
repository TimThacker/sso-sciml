"""Linearisation by automatic differentiation, eigenvalues and participation factors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch


def jacobian(rhs: Callable, x: torch.Tensor, p: dict) -> torch.Tensor:
    """State matrix A = d(rhs)/dx at a single operating point."""
    return torch.func.jacrev(lambda xx: rhs(xx, p))(x)


def newton_equilibrium(
    rhs: Callable, x0: torch.Tensor, p: dict, tol: float = 1e-12, max_iter: int = 25
) -> torch.Tensor:
    """Polish an initial guess so that rhs(x, p) = 0."""
    x = x0.clone()
    for _ in range(max_iter):
        f = rhs(x, p)
        if f.abs().max() < tol:
            break
        x = x - torch.linalg.solve(jacobian(rhs, x, p), f)
    return x


@dataclass(frozen=True)
class Modes:
    eigenvalues: np.ndarray  # (n,) complex: real part 1/s, imaginary part rad/s
    right: np.ndarray  # (n, n) right eigenvectors as columns
    participation: np.ndarray  # (n_state, n_mode), each column sums to 1

    @property
    def freq_hz(self) -> np.ndarray:
        return self.eigenvalues.imag / (2 * np.pi)

    @property
    def sigma(self) -> np.ndarray:
        return self.eigenvalues.real


def modes(a: torch.Tensor | np.ndarray) -> Modes:
    """Eigen-decomposition on the CPU (see docs/steps/00-setup.md for why not the GPU)."""
    a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    lam, vl, vr = scipy.linalg.eig(a_np, left=True, right=True)
    part = np.abs(vl.conj() * vr)
    part = part / part.sum(axis=0, keepdims=True)
    return Modes(lam, vr, part)


def quasi_static(a: torch.Tensor | np.ndarray, fast: list[int]) -> tuple[np.ndarray, list[int]]:
    """State matrix with the `fast` states made algebraic (their derivatives set to zero).

    Returns the reduced matrix and the indices of the states it keeps, in order. This is
    the linear form of dropping stator and network transients.
    """
    a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    slow = [k for k in range(a_np.shape[0]) if k not in set(fast)]
    a_ss, a_sf = a_np[np.ix_(slow, slow)], a_np[np.ix_(slow, fast)]
    a_fs, a_ff = a_np[np.ix_(fast, slow)], a_np[np.ix_(fast, fast)]
    return a_ss - a_sf @ np.linalg.solve(a_ff, a_fs), slow
