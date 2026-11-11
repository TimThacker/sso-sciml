"""Synchronous machine equivalent circuits (two rotor circuits per axis)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisCircuit:
    """One axis of the machine: stator leakage, mutual, and two rotor circuits.

    All values in per unit on the machine base. Rotor circuit 1 is the field (d axis)
    or the first damper (q axis); rotor circuit 2 is the remaining damper.
    """

    x_sync: float  # synchronous reactance Xd or Xq
    x_leak: float  # stator leakage Xl
    x_mut: float  # mutual reactance Xad or Xaq
    x_r1: float  # rotor circuit 1 leakage
    x_r2: float  # rotor circuit 2 leakage
    r_r1: float  # rotor circuit 1 resistance
    r_r2: float  # rotor circuit 2 resistance

    def rotor_inductance(self) -> np.ndarray:
        m = self.x_mut
        return np.array([[m + self.x_r1, m], [m, m + self.x_r2]])

    def rotor_resistance(self) -> np.ndarray:
        return np.diag([self.r_r1, self.r_r2])

    def operational_reactance(self, s: complex) -> complex:
        """X(s): stator flux per unit stator current with the rotor circuits shorted.

        s is the per-unit complex frequency seen from the rotor (j times slip frequency for
        a steady oscillation). X(0) is the synchronous reactance; X(inf) is the
        subtransient reactance.
        """
        ones = np.ones(2)
        a = s * self.rotor_inductance() + self.rotor_resistance()
        return self.x_sync - s * self.x_mut**2 * (ones @ np.linalg.solve(a, ones))


def axis_from_standard(
    x_sync: float,
    x_trans: float,
    x_sub: float,
    x_leak: float,
    t_open_trans: float,
    t_open_sub: float,
    f0: float,
) -> AxisCircuit:
    """Classical conversion from standard parameters to equivalent-circuit parameters.

    Uses the classical definitions of the transient and subtransient quantities
    (Kundur, Power System Stability and Control, section 4.2): each open-circuit time
    constant is attributed to a single rotor circuit. The reactances are reproduced
    exactly; the time constants are reproduced only approximately.
    """
    wb = 2 * np.pi * f0
    x_mut = x_sync - x_leak
    a = x_trans - x_leak
    x_r1 = x_mut * a / (x_mut - a)
    b = x_sub - x_leak
    x_r2 = b * x_mut * x_r1 / (x_mut * x_r1 - b * (x_mut + x_r1))
    r_r1 = (x_mut + x_r1) / (wb * t_open_trans)
    r_r2 = (x_r2 + x_mut * x_r1 / (x_mut + x_r1)) / (wb * t_open_sub)
    return AxisCircuit(x_sync, x_leak, x_mut, x_r1, x_r2, r_r1, r_r2)
