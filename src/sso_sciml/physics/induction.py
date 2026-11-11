"""System B1: fixed-speed induction generator on a series-compensated line.

A squirrel-cage machine with a single rotor circuit feeds an infinite bus through a
series R, L and C. Stator and network transients are kept. The rotor is round, so the
model is written in the synchronous frame and needs no rotor angle.

State vector (7):
    0  psi_sD   stator flux including the external series inductance, D axis
    1  psi_sQ   same, Q axis
    2  psi_rD   rotor flux, D axis
    3  psi_rQ   rotor flux, Q axis
    4  vc_D     series capacitor voltage, D axis
    5  vc_Q     series capacitor voltage, Q axis
    6  w_r      rotor speed, pu

Every machine and network quantity is an entry of the parameter dictionary, so the
right-hand side is differentiable with respect to all of them and can be batched over
them. See docs/EQUATIONS.md section 12 for the derivation and sign conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import brentq

N_STATE = 7
N_ELECTRICAL = 6
W_R = 6
STATE_NAMES = ("psi_sD", "psi_sQ", "psi_rD", "psi_rQ", "vc_D", "vc_Q", "w_r")


@dataclass(frozen=True)
class IGData:
    """Machine data in per unit on the plant rating; network data on the same base.

    Machine values are representative of a 2 MW class wind-turbine induction machine.
    The network is the First Benchmark Model line (0.03 + j0.70 pu on 892.4 MVA) seen
    from a plant of `plant_mva`. Both sets were entered from memory of secondary sources;
    see assumption A12 in docs/ASSUMPTIONS.md.
    """

    f0: float = 60.0
    r_s: float = 0.00488
    x_ls: float = 0.09241
    x_m: float = 3.95279
    r_r: float = 0.00549
    x_lr: float = 0.09955
    h: float = 3.5  # lumped turbine and generator inertia constant, s
    plant_mva: float = 500.0
    line_mva: float = 892.4
    r_line_pu: float = 0.03  # on the line base
    x_line_pu: float = 0.70  # on the line base

    @property
    def wb(self) -> float:
        return 2 * np.pi * self.f0

    @property
    def r_e(self) -> float:
        return self.r_line_pu * self.plant_mva / self.line_mva

    @property
    def x_e(self) -> float:
        return self.x_line_pu * self.plant_mva / self.line_mva


def _t(v) -> torch.Tensor:
    return torch.as_tensor(v, dtype=torch.float64)


@dataclass
class InductionGenerator:
    data: IGData = field(default_factory=IGData)

    # ------------------------------------------------------------------ parameters
    def params(self, x_c, v_b=1.0, t_m=0.0, **overrides) -> dict:
        """Parameter dictionary for `rhs`. Every entry may carry leading batch dimensions.

        Keyword overrides replace any of r_e, x_e, r_s, x_ls, x_m, r_r, x_lr, h.
        """
        dat = self.data
        base = {
            "r_e": dat.r_e,
            "x_e": dat.x_e,
            "r_s": dat.r_s,
            "x_ls": dat.x_ls,
            "x_m": dat.x_m,
            "r_r": dat.r_r,
            "x_lr": dat.x_lr,
            "h": dat.h,
        }
        unknown = set(overrides) - set(base)
        if unknown:
            raise KeyError(f"unknown parameters: {sorted(unknown)}")
        base.update({k: v for k, v in overrides.items() if v is not None})
        p = {k: _t(v) for k, v in base.items()}
        p.update({"x_c": _t(x_c), "v_b": _t(v_b), "t_m": _t(t_m)})
        return p

    def currents(self, x: torch.Tensor, p: dict) -> dict:
        """Stator current (positive out of the machine) and rotor current from fluxes."""
        x_ss = p["x_ls"] + p["x_m"] + p["x_e"]
        x_rr = p["x_lr"] + p["x_m"]
        x_m = p["x_m"]
        det = x_m * x_m - x_ss * x_rr
        return {
            "i_D": (x_rr * x[..., 0] - x_m * x[..., 2]) / det,
            "i_Q": (x_rr * x[..., 1] - x_m * x[..., 3]) / det,
            "ir_D": (x_m * x[..., 0] - x_ss * x[..., 2]) / det,
            "ir_Q": (x_m * x[..., 1] - x_ss * x[..., 3]) / det,
        }

    def torque(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        """Air-gap torque, pu, positive when it decelerates the rotor (generating)."""
        c = self.currents(x, p)
        return x[..., 0] * c["i_Q"] - x[..., 1] * c["i_D"]

    # ------------------------------------------------------------------ dynamics
    def rhs(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        """dx/dt in seconds for state x of shape (..., 7)."""
        wb = self.data.wb
        c = self.currents(x, p)
        i_d, i_q, ir_d, ir_q = c["i_D"], c["i_Q"], c["ir_D"], c["ir_Q"]
        ps_d, ps_q, pr_d, pr_q = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        vc_d, vc_q, w_r = x[..., 4], x[..., 5], x[..., 6]
        r = p["r_s"] + p["r_e"]
        slip = 1.0 - w_r

        dps_d = wb * (p["v_b"] + vc_d + r * i_d + ps_q)
        dps_q = wb * (vc_q + r * i_q - ps_d)
        dpr_d = wb * (-p["r_r"] * ir_d + slip * pr_q)
        dpr_q = wb * (-p["r_r"] * ir_q - slip * pr_d)
        dvc_d = wb * (p["x_c"] * i_d + vc_q)
        dvc_q = wb * (p["x_c"] * i_q - vc_d)
        t_e = ps_d * i_q - ps_q * i_d
        dw = (p["t_m"] - t_e) / (2.0 * p["h"])
        return torch.stack([dps_d, dps_q, dpr_d, dpr_q, dvc_d, dvc_q, dw], dim=-1)

    def terminal_voltage(self, x: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Stator terminal voltage (D, Q), including the L di/dt drop of the line."""
        wb = self.data.wb
        f = self.rhs(x, p)
        c = self.currents(x, p)
        x_ss = p["x_ls"] + p["x_m"] + p["x_e"]
        x_rr = p["x_lr"] + p["x_m"]
        det = p["x_m"] * p["x_m"] - x_ss * x_rr
        di_d = (x_rr * f[..., 0] - p["x_m"] * f[..., 2]) / det
        di_q = (x_rr * f[..., 1] - p["x_m"] * f[..., 3]) / det
        v_d = p["v_b"] + x[..., 4] + p["r_e"] * c["i_D"] + p["x_e"] * (di_d / wb - c["i_Q"])
        v_q = x[..., 5] + p["r_e"] * c["i_Q"] + p["x_e"] * (di_q / wb + c["i_D"])
        return v_d, v_q

    # ------------------------------------------------------------------ steady state
    def _phasors(self, slip: float, x_c: float, v_b: float, q: dict) -> tuple:
        """Steady-state stator current (out of the machine) and rotor current at a slip."""
        z_rotor = q["r_r"] / slip + 1j * q["x_lr"]
        z_mag = 1j * q["x_m"]
        z_tot = (
            q["r_e"]
            + q["r_s"]
            + 1j * (q["x_e"] - x_c + q["x_ls"])
            + z_mag * z_rotor / (z_mag + z_rotor)
        )
        i_motor = v_b / z_tot
        i_rotor = -i_motor * z_mag / (z_mag + z_rotor)
        return -i_motor, i_rotor

    def equilibrium(
        self,
        x_c: float,
        p_gen: float | None = 0.9,
        w_r: float | None = None,
        v_b: float = 1.0,
        **overrides,
    ) -> tuple[torch.Tensor, dict]:
        """Steady state at a given rotor speed, or at a given power into the infinite bus.

        With `w_r` set, the mechanical torque that holds that speed is solved for. With
        `p_gen` set, the slip on the stable side of pull-out is found first.
        """
        p0 = self.params(x_c, v_b=v_b, **overrides)
        q = {k: float(v) for k, v in p0.items()}

        def bus_power(slip: float) -> float:
            i, _ = self._phasors(slip, x_c, v_b, q)
            return (v_b * np.conj(i)).real

        if w_r is None:
            # Power rises with |slip| up to pull-out; search between zero and the peak.
            grid = -np.geomspace(1e-5, 0.2, 400)
            pw = np.array([bus_power(s) for s in grid])
            k = int(np.argmax(pw))
            if pw[k] < p_gen:
                raise ValueError(f"p_gen = {p_gen} exceeds the pull-out power {pw[k]:.3f} pu")
            slip = brentq(lambda s: bus_power(s) - p_gen, grid[0], grid[k], xtol=1e-15)
        else:
            slip = 1.0 - w_r
            if slip == 0.0:
                raise ValueError("w_r = 1 is the no-load point; pick a speed off synchronism")

        i, i_r = self._phasors(slip, x_c, v_b, q)
        psi_s = -(q["x_ls"] + q["x_m"] + q["x_e"]) * i + q["x_m"] * i_r
        psi_r = -q["x_m"] * i + (q["x_lr"] + q["x_m"]) * i_r
        v_c = -1j * x_c * i
        t_e = (np.conj(psi_s) * i).imag

        x = np.array(
            [psi_s.real, psi_s.imag, psi_r.real, psi_r.imag, v_c.real, v_c.imag, 1.0 - slip]
        )
        p0["t_m"] = _t(t_e)
        return torch.as_tensor(x, dtype=torch.float64), p0
