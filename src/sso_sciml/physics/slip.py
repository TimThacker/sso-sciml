"""Induction generator effect from the machine equivalent circuit.

A stator current at a sub-synchronous frequency f_e produces an mmf that rotates slower
than the rotor. The rotor circuits therefore see slip

    s = (f_e - f_r) / f_e  < 0,

and their resistance, reflected to the stator, is negative. If that negative resistance
is larger than the resistance of the network at the electrical resonance, the resonance
grows by itself. Nothing here uses the dynamic model: it is the hand calculation that the
dynamic model has to agree with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .machine import AxisCircuit


def machine_impedance(
    d: AxisCircuit, q: AxisCircuit, s: complex, w_r: float = 1.0, ra: float = 0.0
) -> complex:
    """Stator impedance at per-unit complex frequency s (stator frame), rotor at speed w_r.

    The d and q operational reactances are averaged, which neglects the coupling between
    the sub- and super-synchronous components that saliency introduces.
    """
    s_rotor = s - 1j * w_r
    x_avg = 0.5 * (d.operational_reactance(s_rotor) + q.operational_reactance(s_rotor))
    return ra + s * x_avg


def total_impedance(
    d: AxisCircuit,
    q: AxisCircuit,
    s: complex,
    r_e: float,
    x_e: float,
    x_c: float,
    w_r: float = 1.0,
    ra: float = 0.0,
) -> complex:
    """Loop impedance: series R, L, C plus the machine."""
    return r_e + s * x_e + x_c / s + machine_impedance(d, q, s, w_r, ra)


@dataclass(frozen=True)
class SlipResult:
    f_er: float  # electrical resonance frequency in the stator frame, Hz
    slip: float  # rotor slip at that frequency
    r_machine: float  # machine resistance at resonance, pu (negative below synchronous speed)
    r_net: float  # loop resistance at resonance, pu
    sigma_hand: float  # growth rate from R_net and the reactance slope, 1/s
    sigma_circuit: float  # growth rate from the complex root of the loop impedance, 1/s
    f_circuit: float  # frequency of that root in the stator frame, Hz

    @property
    def self_excited(self) -> bool:
        return self.r_net < 0.0


def analyse(
    d: AxisCircuit,
    q: AxisCircuit,
    r_e: float,
    x_e: float,
    x_c: float,
    f0: float = 60.0,
    w_r: float = 1.0,
    ra: float = 0.0,
) -> SlipResult:
    """Resonant frequency, slip, equivalent resistances, and the growth-rate estimates."""
    wb = 2 * np.pi * f0

    def z(w: float) -> complex:
        return total_impedance(d, q, 1j * w, r_e, x_e, x_c, w_r, ra)

    w_er = brentq(lambda w: z(w).imag, 1e-3, w_r - 1e-6)
    z_res = z(w_er)
    h = 1e-6
    dz_dw = (z(w_er + h) - z(w_er - h)) / (2 * h)
    r_machine = machine_impedance(d, q, 1j * w_er, w_r, ra).real
    sigma_hand = -wb * z_res.real / dz_dw.imag

    # Complex root of Z(s) = 0 by Newton iteration, started on the imaginary axis.
    s = 1j * w_er
    for _ in range(50):
        f = total_impedance(d, q, s, r_e, x_e, x_c, w_r, ra)
        df = (
            total_impedance(d, q, s + h, r_e, x_e, x_c, w_r, ra)
            - total_impedance(d, q, s - h, r_e, x_e, x_c, w_r, ra)
        ) / (2 * h)
        step = f / df
        s = s - step
        if abs(step) < 1e-13:
            break
    return SlipResult(
        f_er=w_er * f0,
        slip=(w_er - w_r) / w_er,
        r_machine=r_machine,
        r_net=z_res.real,
        sigma_hand=sigma_hand,
        sigma_circuit=s.real * wb,
        f_circuit=s.imag * f0,
    )
