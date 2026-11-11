"""IEEE First Benchmark Model for sub-synchronous resonance.

A synchronous generator with a six-mass shaft feeds an infinite bus through a
series-compensated line. Stator and network transients are kept, which is what lets the
sub-synchronous network mode exist in the model.

State vector (20):
    0  psi_d    stator d flux including the external series inductance
    1  psi_q    stator q flux including the external series inductance
    2  psi_fd   field flux
    3  psi_1d   d-axis damper flux
    4  psi_1q   q-axis damper 1 flux
    5  psi_2q   q-axis damper 2 flux
    6  vc_D     series capacitor voltage, synchronous frame D axis
    7  vc_Q     series capacitor voltage, synchronous frame Q axis
    8..13   delta_i   mass angles, electrical rad (HP, IP, LPA, LPB, GEN, EXC)
    14..19  dw_i      mass speed deviations, pu

See docs/EQUATIONS.md for the derivation and sign conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .machine import AxisCircuit, axis_from_standard

MASSES = ("HP", "IP", "LPA", "LPB", "GEN", "EXC")
GEN = 4
N_STATE = 20
N_ELECTRICAL = 8
STATE_NAMES = (
    "psi_d",
    "psi_q",
    "psi_fd",
    "psi_1d",
    "psi_1q",
    "psi_2q",
    "vc_D",
    "vc_Q",
    *(f"delta_{m}" for m in MASSES),
    *(f"dw_{m}" for m in MASSES),
)


@dataclass(frozen=True)
class FBMData:
    """Benchmark data in per unit on the generator base (892.4 MVA, 60 Hz).

    Values are as commonly tabulated for the benchmark in the SSR literature. They were
    entered from memory of secondary sources and have not been checked line by line
    against the 1977 paper; see docs/steps/01-solver-core-fbm.md.
    """

    f0: float = 60.0
    # Network
    r_t: float = 0.01
    x_t: float = 0.14
    r_line: float = 0.02
    x_line: float = 0.50
    x_sys: float = 0.06
    # Generator
    xd: float = 1.79
    xq: float = 1.71
    xd_p: float = 0.169
    xq_p: float = 0.228
    xd_pp: float = 0.135
    xq_pp: float = 0.200
    xl: float = 0.13
    ra: float = 0.0
    td0_p: float = 4.3
    td0_pp: float = 0.032
    tq0_p: float = 0.85
    tq0_pp: float = 0.05
    # Shaft: HP, IP, LPA, LPB, GEN, EXC
    inertia_h: tuple[float, ...] = (0.092897, 0.155589, 0.858670, 0.884215, 0.868495, 0.0342165)
    stiffness_k: tuple[float, ...] = (19.303, 34.929, 52.038, 70.858, 2.822)
    damping_d: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    torque_fraction: tuple[float, ...] = (0.30, 0.26, 0.22, 0.22)

    @property
    def wb(self) -> float:
        return 2 * np.pi * self.f0

    @property
    def r_e(self) -> float:
        return self.r_t + self.r_line

    @property
    def x_e(self) -> float:
        return self.x_t + self.x_line + self.x_sys

    def d_axis(self) -> AxisCircuit:
        return axis_from_standard(
            self.xd, self.xd_p, self.xd_pp, self.xl, self.td0_p, self.td0_pp, self.f0
        )

    def q_axis(self) -> AxisCircuit:
        return axis_from_standard(
            self.xq, self.xq_p, self.xq_pp, self.xl, self.tq0_p, self.tq0_pp, self.f0
        )


def _t(v) -> torch.Tensor:
    return torch.as_tensor(v, dtype=torch.float64)


@dataclass
class FBM:
    data: FBMData = field(default_factory=FBMData)

    def __post_init__(self) -> None:
        self.d = self.data.d_axis()
        self.q = self.data.q_axis()

    # ------------------------------------------------------------------ parameters
    def params(self, x_c, v_b=1.0, v_fd=0.0, t_m=0.0, x_e=None, r_e=None) -> dict:
        """Parameter dictionary for `rhs`. Every entry may carry leading batch dimensions."""
        x_e_t = _t(self.data.x_e if x_e is None else x_e)
        return {
            "x_c": _t(x_c),
            "x_e": x_e_t,
            "r_e": _t(self.data.r_e if r_e is None else r_e),
            "v_b": _t(v_b),
            "v_fd": _t(v_fd),
            "t_m": _t(t_m),
            # Inverse inductance matrices, so `rhs` needs only matrix products.
            "ld_inv": torch.linalg.inv(self._axis_matrix(self.d, x_e_t)),
            "lq_inv": torch.linalg.inv(self._axis_matrix(self.q, x_e_t)),
        }

    def _axis_matrix(self, ax: AxisCircuit, x_e: torch.Tensor) -> torch.Tensor:
        """Flux = L @ current for [stator, rotor1, rotor2], generator convention on the stator."""
        m = ax.x_mut + torch.zeros_like(x_e)
        return torch.stack(
            [
                torch.stack([-(ax.x_sync + x_e), m, m], dim=-1),
                torch.stack([-m, m + ax.x_r1, m], dim=-1),
                torch.stack([-m, m, m + ax.x_r2], dim=-1),
            ],
            dim=-2,
        )

    def currents(self, x: torch.Tensor, p: dict) -> dict:
        """Winding currents from fluxes."""
        i_dax = (p["ld_inv"] @ x[..., [0, 2, 3]].unsqueeze(-1)).squeeze(-1)
        i_qax = (p["lq_inv"] @ x[..., [1, 4, 5]].unsqueeze(-1)).squeeze(-1)
        return {
            "i_d": i_dax[..., 0],
            "i_fd": i_dax[..., 1],
            "i_1d": i_dax[..., 2],
            "i_q": i_qax[..., 0],
            "i_1q": i_qax[..., 1],
            "i_2q": i_qax[..., 2],
        }

    def torque(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        """Air-gap torque, pu, positive when it decelerates the generator."""
        c = self.currents(x, p)
        return x[..., 0] * c["i_q"] - x[..., 1] * c["i_d"]

    # ------------------------------------------------------------------ dynamics
    def rhs(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        """dx/dt in seconds for state x of shape (..., 20)."""
        dat = self.data
        wb = dat.wb
        c = self.currents(x, p)
        i_d, i_q = c["i_d"], c["i_q"]
        psi_d, psi_q = x[..., 0], x[..., 1]
        vc_d, vc_q = x[..., 6], x[..., 7]
        delta = x[..., 8:14]
        dw = x[..., 14:20]

        dg = delta[..., GEN]
        w = 1.0 + dw[..., GEN]
        sin, cos = torch.sin(dg), torch.cos(dg)

        # Voltage behind the series R-L, moved into the rotor frame.
        a = vc_d + p["v_b"]
        b = vc_q
        v_d = a * sin - b * cos
        v_q = a * cos + b * sin
        r = dat.ra + p["r_e"]

        dpsi_d = wb * (v_d + r * i_d + w * psi_q)
        dpsi_q = wb * (v_q + r * i_q - w * psi_d)
        dpsi_fd = wb * (p["v_fd"] - self.d.r_r1 * c["i_fd"])
        dpsi_1d = -wb * self.d.r_r2 * c["i_1d"]
        dpsi_1q = -wb * self.q.r_r1 * c["i_1q"]
        dpsi_2q = -wb * self.q.r_r2 * c["i_2q"]

        # Line current in the synchronous frame charges the capacitor.
        i_big_d = i_d * sin + i_q * cos
        i_big_q = -i_d * cos + i_q * sin
        dvc_d = wb * (p["x_c"] * i_big_d + vc_q)
        dvc_q = wb * (p["x_c"] * i_big_q - vc_d)

        t_e = psi_d * i_q - psi_q * i_d
        frac = _t(dat.torque_fraction).to(x.device)
        zero = torch.zeros_like(t_e).unsqueeze(-1)
        t_turb = (p["t_m"] + torch.zeros_like(t_e)).unsqueeze(-1) * frac
        applied = torch.cat([t_turb, -t_e.unsqueeze(-1), zero], dim=-1)

        k = _t(dat.stiffness_k).to(x.device)
        section = k * (delta[..., :-1] - delta[..., 1:])  # torque carried by each shaft section
        net = applied - _t(dat.damping_d).to(x.device) * dw
        net = net - torch.cat([section, zero], dim=-1) + torch.cat([zero, section], dim=-1)
        ddw = net / (2.0 * _t(dat.inertia_h).to(x.device))
        ddelta = wb * dw

        elec = torch.stack(
            [dpsi_d, dpsi_q, dpsi_fd, dpsi_1d, dpsi_1q, dpsi_2q, dvc_d, dvc_q], dim=-1
        )
        return torch.cat([elec, ddelta, ddw], dim=-1)

    # ------------------------------------------------------------------ steady state
    def equilibrium(
        self,
        x_c: float,
        p_gen: float = 0.9,
        pf: float = 0.9,
        v_t: float = 1.0,
        x_e: float | None = None,
        r_e: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Steady state for a given generator terminal condition (lagging power factor).

        Returns the state and the parameter dictionary (infinite-bus voltage, field voltage
        and mechanical torque are solved for).
        """
        dat = self.data
        x_e = dat.x_e if x_e is None else x_e
        r_e = dat.r_e if r_e is None else r_e
        q_gen = p_gen * np.tan(np.arccos(pf))
        i_ph = np.conj((p_gen + 1j * q_gen) / v_t)
        vb_ph = v_t - (r_e + 1j * (x_e - x_c)) * i_ph
        rot = np.exp(-1j * np.angle(vb_ph))  # put the infinite bus on the D axis
        i_ph, vt_ph, v_b = i_ph * rot, v_t * rot, abs(vb_ph)
        vc_ph = -1j * x_c * i_ph

        e_q = vt_ph + (dat.ra + 1j * dat.xq) * i_ph
        delta = np.angle(e_q)
        to_rotor = np.exp(-1j * (delta - np.pi / 2))
        i_dq, v_dq = i_ph * to_rotor, vt_ph * to_rotor
        i_d, i_q, v_d, v_q = i_dq.real, i_dq.imag, v_dq.real, v_dq.imag

        psi_d = v_q + dat.ra * i_q
        psi_q = -(v_d + dat.ra * i_d)
        i_fd = (psi_d + dat.xd * i_d) / self.d.x_mut
        t_e = psi_d * i_q - psi_q * i_d

        x = np.zeros(N_STATE)
        x[0] = psi_d - x_e * i_d
        x[1] = psi_q - x_e * i_q
        x[2] = -self.d.x_mut * i_d + (self.d.x_mut + self.d.x_r1) * i_fd
        x[3] = -self.d.x_mut * i_d + self.d.x_mut * i_fd
        x[4] = -self.q.x_mut * i_q
        x[5] = -self.q.x_mut * i_q
        x[6], x[7] = vc_ph.real, vc_ph.imag

        # Shaft twist: each section carries the turbine torque applied upstream of it.
        carried = np.cumsum(np.array(dat.torque_fraction) * t_e)
        angles = np.zeros(6)
        angles[GEN] = delta
        for i in range(GEN - 1, -1, -1):
            angles[i] = angles[i + 1] + carried[i] / dat.stiffness_k[i]
        angles[5] = delta
        x[8:14] = angles

        p = self.params(x_c, v_b=v_b, v_fd=self.d.r_r1 * i_fd, t_m=t_e, x_e=x_e, r_e=r_e)
        return torch.as_tensor(x, dtype=torch.float64), p


def stack_cases(cases: list[tuple[torch.Tensor, dict]], device: str = "cpu") -> tuple:
    """Stack single operating points into one batch: states (n, 20) and batched parameters."""
    x = torch.stack([c[0] for c in cases]).to(device)
    p = {k: torch.stack([c[1][k] for c in cases]).to(device) for k in cases[0][1]}
    return x, p
