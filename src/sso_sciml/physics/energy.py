"""Energy accounting for an oscillation, from the time-domain waveform.

Take the deviation of the stator current and terminal voltage from the steady state. The
product Re(dv * conj(di)) is the power the machine delivers through the oscillation
itself. If the machine acts as a resistance R_machine at the oscillation frequency, that
power is -R_machine * |di|^2, so a negative resistance shows up as energy flowing out of
the machine and into the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid


@dataclass(frozen=True)
class EnergyAccount:
    """Running integrals over the trajectory; the last entry of each is the total."""

    t: np.ndarray  # s
    e_machine: np.ndarray  # energy delivered by the machine through the oscillation, pu * s
    i_squared: np.ndarray  # integral of |di|^2, pu^2 * s
    r_e: float  # series resistance of the network, pu

    @property
    def e_network_loss(self) -> np.ndarray:
        """Energy dissipated in the series resistance, pu * s."""
        return self.r_e * self.i_squared

    @property
    def r_effective(self) -> float:
        """The machine resistance the whole waveform implies, pu."""
        return float(-self.e_machine[-1] / self.i_squared[-1])


def account(model, traj: torch.Tensor, x0: torch.Tensor, p: dict, dt: float) -> EnergyAccount:
    """Integrate oscillation power over a trajectory sampled every `dt` seconds.

    `model` must provide `currents` (with keys i_D, i_Q) and `terminal_voltage`.
    """

    def phasors(x):
        c = model.currents(x, p)
        v_d, v_q = model.terminal_voltage(x, p)
        return torch.complex(c["i_D"], c["i_Q"]).numpy(), torch.complex(v_d, v_q).numpy()

    i0, v0 = phasors(x0)
    i, v = phasors(traj)
    di, dv = i - i0, v - v0
    return EnergyAccount(
        t=np.arange(len(di)) * dt,
        e_machine=cumulative_trapezoid((dv * np.conj(di)).real, dx=dt, initial=0.0),
        i_squared=cumulative_trapezoid(np.abs(di) ** 2, dx=dt, initial=0.0),
        r_e=float(p["r_e"]),
    )
