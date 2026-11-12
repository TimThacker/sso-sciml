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

from collections.abc import Callable
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


def induction_impedance(
    s: complex, r_s: float, x_ls: float, x_m: float, r_r: float, x_lr: float, w_r: float = 1.0
) -> complex:
    """Stator impedance of a single-cage induction machine at complex frequency s.

    The rotor branch is R_r / slip + s X_lr with slip = (s - j w_r) / s. For a round rotor
    this is exact: no averaging of axes is involved.
    """
    slip = (s - 1j * w_r) / s
    z_rotor = r_r / slip + s * x_lr
    z_mag = s * x_m
    return r_s + s * x_ls + z_mag * z_rotor / (z_mag + z_rotor)


def dfig_impedance(
    s: complex,
    r_s: float,
    x_ls: float,
    x_m: float,
    r_r: float,
    x_lr: float,
    w_r: float = 1.0,
    kp: float = 0.0,
    ki: float = 0.0,
    f0: float = 60.0,
) -> complex:
    """Stator impedance of a doubly-fed machine whose rotor current is PI-controlled.

    A controller acting on rotor current in the synchronous frame puts Kp + Ki / S in
    series with the rotor resistance, where S = wb (s - j) is the synchronous-frame
    frequency in 1/s. Kp is plain added rotor resistance. With Kp = Ki = 0 this is
    `induction_impedance`.
    """
    wb = 2 * np.pi * f0
    r_eff = r_r + kp + (ki / (wb * (s - 1j)) if ki else 0.0)
    slip = (s - 1j * w_r) / s
    z_rotor = r_eff / slip + s * x_lr
    z_mag = s * x_m
    return r_s + s * x_ls + z_mag * z_rotor / (z_mag + z_rotor)


def analyse_loop(
    z_machine: Callable[[complex], complex],
    r_e: float,
    x_e: float,
    x_c: float,
    f0: float = 60.0,
    w_r: float = 1.0,
    w_max: float | None = None,
    z_network: Callable[[complex], complex] | None = None,
) -> SlipResult:
    """Slip analysis of a series R-L-C loop closed through any machine impedance z_machine(s).

    The resonance is searched below `w_max` (default: just below rotor speed, where slip
    is negative). Raises ValueError if the loop reactance does not cross zero there.
    With `z_network` given, it replaces the series R-L-C and r_e, x_e, x_c are ignored.
    """
    wb = 2 * np.pi * f0

    def z_loop(s: complex) -> complex:
        if z_network is not None:
            return z_network(s) + z_machine(s)
        return r_e + s * x_e + x_c / s + z_machine(s)

    # The loop reactance can cross zero more than once (a controller with integral action
    # adds crossings near synchronous frequency). The network resonance is the lowest
    # crossing from capacitive to inductive.
    hi = (w_r - 1e-6) if w_max is None else w_max
    grid = np.linspace(1e-3, hi, 600)
    x_grid = np.array([z_loop(1j * w).imag for w in grid])
    up = np.where((x_grid[:-1] < 0) & (x_grid[1:] >= 0))[0]
    if up.size == 0:
        raise ValueError("loop reactance does not cross zero below the search limit")
    w_er = brentq(lambda w: z_loop(1j * w).imag, grid[up[0]], grid[up[0] + 1], xtol=1e-14)
    z_res = z_loop(1j * w_er)
    h = 1e-6
    dz_dw = (z_loop(1j * (w_er + h)) - z_loop(1j * (w_er - h))) / (2 * h)
    sigma_hand = -wb * z_res.real / dz_dw.imag

    # Complex root of Z(s) = 0 by Newton iteration, started on the imaginary axis.
    s = 1j * w_er
    for _ in range(50):
        step = z_loop(s) / ((z_loop(s + h) - z_loop(s - h)) / (2 * h))
        s = s - step
        if abs(step) < 1e-13:
            break
    return SlipResult(
        f_er=w_er * f0,
        slip=(w_er - w_r) / w_er,
        r_machine=z_machine(1j * w_er).real,
        r_net=z_res.real,
        sigma_hand=sigma_hand,
        sigma_circuit=s.real * wb,
        f_circuit=s.imag * f0,
    )


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
    """Slip analysis for a synchronous machine given by its two axis circuits."""
    return analyse_loop(lambda s: machine_impedance(d, q, s, w_r, ra), r_e, x_e, x_c, f0, w_r)


def analyse_induction(
    r_s: float,
    x_ls: float,
    x_m: float,
    r_r: float,
    x_lr: float,
    r_e: float,
    x_e: float,
    x_c: float,
    f0: float = 60.0,
    w_r: float = 1.0,
    w_max: float | None = None,
) -> SlipResult:
    """Slip analysis for a single-cage induction machine."""
    return analyse_loop(
        lambda s: induction_impedance(s, r_s, x_ls, x_m, r_r, x_lr, w_r),
        r_e,
        x_e,
        x_c,
        f0,
        w_r,
        w_max,
    )
