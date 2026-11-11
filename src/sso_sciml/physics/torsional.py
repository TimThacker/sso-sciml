"""Torsional interaction from the electrical damping torque.

When the generator rotor oscillates at a torsional frequency f_n, the stator sees currents
at f0 - f_n and f0 + f_n. If f0 - f_n sits on the electrical resonance, the sub-synchronous
current is large and the torque it produces is in phase with rotor speed in the direction
that feeds the oscillation: negative electrical damping.

Here the electrical side is reduced to one complex number per frequency, the torque
coefficient K_e(s) = dT_e / d(delta_gen), and the shaft is reduced to one modal mass per
torsional mode. The growth rate of mode n follows from a first-order perturbation.
"""

from __future__ import annotations

import numpy as np

from .fbm import GEN, N_ELECTRICAL

_DELTA_GEN = N_ELECTRICAL + GEN
_DW_GEN = N_ELECTRICAL + 6 + GEN


def torque_coefficient(a: np.ndarray, h_gen: float, wb: float, s: complex) -> complex:
    """K_e(s): air-gap torque response to an imposed generator angle delta_gen(s).

    a is the full 20-state matrix. Rotor speed follows the angle, dw = s * delta / wb.
    """
    a_ee = a[:N_ELECTRICAL, :N_ELECTRICAL]
    b = a[:N_ELECTRICAL, _DELTA_GEN] + (s / wb) * a[:N_ELECTRICAL, _DW_GEN]
    c = -2.0 * h_gen * a[_DW_GEN, :N_ELECTRICAL]  # dT_e / d(electrical states)
    return complex(c @ np.linalg.solve(s * np.eye(N_ELECTRICAL) - a_ee, b))


def damping_torque(a: np.ndarray, h_gen: float, wb: float, freq_hz: float) -> float:
    """Electrical damping coefficient D_e at freq_hz, pu torque per pu speed."""
    w = 2 * np.pi * freq_hz
    return torque_coefficient(a, h_gen, wb, 1j * w).imag * wb / w


def predicted_growth(
    a: np.ndarray,
    inertia_h: np.ndarray,
    shape: np.ndarray,
    freq_hz: float,
    wb: float,
) -> float:
    """Growth rate (1/s) of one torsional mode from the electrical damping torque.

    sigma_n = -D_e(f_n) * phi_gen^2 / (4 * H_n), with modal inertia H_n = sum_i H_i phi_i^2.
    """
    h_modal = float(np.sum(np.asarray(inertia_h) * shape**2))
    d_e = damping_torque(a, float(inertia_h[GEN]), wb, freq_hz)
    return -d_e * shape[GEN] ** 2 / (4.0 * h_modal)
