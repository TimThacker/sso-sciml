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


@dataclass(frozen=True)
class EnergyFlows:
    """Where the energy of an oscillation comes from and goes, totals over the run, pu * s."""

    to_network: float  # delivered at the stator terminals
    from_shaft: float  # mechanical energy converted through the air-gap torque
    from_converter: float  # delivered by the rotor converter (negative: it absorbs)
    copper: float  # stator and rotor resistance losses
    stored: float  # increase in magnetic stored energy

    @property
    def closure(self) -> float:
        """Imbalance of shaft + converter = network + copper + stored, relative to network."""
        balance = self.from_shaft + self.from_converter - self.copper - self.stored
        return float((balance - self.to_network) / abs(self.to_network))


def flows(model, traj: torch.Tensor, x0: torch.Tensor, p: dict, dt: float) -> EnergyFlows:
    """Energy budget of the oscillation for a machine whose rotor speed is the parameter w_r.

    Every term is the second-order part: products of deviations from the steady state.
    `model` must also provide `rotor_voltage`.
    """
    wb = model.data.wb

    def quantities(x):
        c = model.currents(x, p)
        v_d, v_q = model.terminal_voltage(x, p)
        vr_d, vr_q = model.rotor_voltage(x, p)
        i = torch.complex(c["i_D"], c["i_Q"]).numpy()
        i_r = torch.complex(c["ir_D"], c["ir_Q"]).numpy()
        psi = torch.complex(x[..., 0], x[..., 1]).numpy()  # stator flux incl. the line
        psi_r = torch.complex(x[..., 2], x[..., 3]).numpy()
        return (
            i,
            i_r,
            torch.complex(v_d, v_q).numpy(),
            torch.complex(vr_d, vr_q).numpy(),
            psi,
            psi_r,
        )

    q0 = quantities(x0)
    di, dir_, dv, dvr, dpsi, dpsi_r = (a - b for a, b in zip(quantities(traj), q0, strict=True))
    w_r, x_e = float(p["w_r"]), float(p["x_e"])

    def integral(y):
        return float(np.trapezoid(y, dx=dt))

    dpsi_s = dpsi + x_e * di  # flux of the machine alone
    stored = (-(np.conj(di) * dpsi_s) + np.conj(dir_) * dpsi_r).real / (2 * wb)
    return EnergyFlows(
        to_network=integral((dv * np.conj(di)).real),
        from_shaft=w_r * integral((np.conj(dpsi) * di).imag),
        from_converter=integral((dvr * np.conj(dir_)).real),
        copper=float(p["r_s"]) * integral(np.abs(di) ** 2)
        + float(p["r_r"]) * integral(np.abs(dir_) ** 2),
        stored=float(stored[-1] - stored[0]),
    )
