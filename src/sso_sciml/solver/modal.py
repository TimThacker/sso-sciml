"""Mode estimation from sampled signals."""

from __future__ import annotations

import numpy as np


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
