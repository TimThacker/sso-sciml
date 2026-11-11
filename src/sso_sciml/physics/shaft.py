"""Lumped multi-mass turbine-generator shaft."""

from __future__ import annotations

import numpy as np


def torsional_modes(
    inertia_h: np.ndarray, stiffness_k: np.ndarray, f0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Natural frequencies (Hz) and mode shapes of a free-free spring-mass chain.

    inertia_h: inertia constants H_i in seconds, one per mass.
    stiffness_k: shaft stiffness between adjacent masses, pu torque per electrical radian.

    Equation of motion with delta in electrical radians and speed in pu:
        (2 H_i / wb) * d2(delta_i)/dt2 = -sum_j K_ij delta_j
    Returns frequencies sorted ascending (the first is the zero-frequency rigid-body mode)
    and mode shapes as columns, each scaled so the largest entry is 1.
    """
    h = np.asarray(inertia_h, dtype=float)
    k = np.asarray(stiffness_k, dtype=float)
    n = h.size
    wb = 2 * np.pi * f0
    kmat = np.zeros((n, n))
    for i, ki in enumerate(k):
        kmat[i, i] += ki
        kmat[i + 1, i + 1] += ki
        kmat[i, i + 1] -= ki
        kmat[i + 1, i] -= ki
    m_inv_sqrt = np.diag(np.sqrt(wb / (2 * h)))
    lam, vec = np.linalg.eigh(m_inv_sqrt @ kmat @ m_inv_sqrt)
    lam = np.clip(lam, 0.0, None)
    shapes = m_inv_sqrt @ vec
    shapes = shapes / np.abs(shapes).max(axis=0)
    return np.sqrt(lam) / (2 * np.pi), shapes
