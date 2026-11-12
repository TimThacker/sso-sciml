"""Mode estimation from sampled signals."""

from __future__ import annotations

import numpy as np


def linear_response(a: np.ndarray, dx0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Deviation from the steady state under dx/dt = A dx, from the modes of A.

    dx(t) = V exp(Lambda t) V^-1 dx0, shape (n_t, n). No time stepping: the cost is one
    eigen-decomposition whatever the length of t.
    """
    lam, v = np.linalg.eig(a)
    c = np.linalg.solve(v, dx0.astype(complex))
    return ((c * np.exp(np.outer(t, lam))) @ v.T).real


def matrix_pencil(y: np.ndarray, dt: float, order: int) -> np.ndarray:
    """Complex poles s_k (real part 1/s, imaginary part rad/s) of `order` damped exponentials
    fitted to the uniformly sampled signal y."""
    y = np.asarray(y, dtype=float)
    n = y.size
    pencil = n // 3
    rows = n - pencil
    hankel = np.lib.stride_tricks.sliding_window_view(y, pencil + 1)[:rows]
    _, _, vh = np.linalg.svd(hankel, full_matrices=False)
    v = vh[:order].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    return np.log(z) / dt


def nearest_pole(poles: np.ndarray, freq_hz: float) -> complex:
    """The fitted pole with positive frequency closest to freq_hz."""
    cand = poles[poles.imag > 0]
    return complex(cand[np.argmin(np.abs(cand.imag / (2 * np.pi) - freq_hz))])
