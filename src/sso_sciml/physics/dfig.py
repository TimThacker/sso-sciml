"""System B2, first stage: doubly-fed induction generator with rotor current control.

The machine and the line are those of `induction.py`. The rotor is no longer shorted: a
converter applies a rotor voltage set by a proportional-integral controller acting on the
rotor current,

    v_r = v_ff + Kp (i_ref - i_r) + Ki * integral(i_ref - i_r) dt,

in the synchronous frame. At this stage the frame is ideal (no phase-locked loop), the
current references are constants (no outer power loops), there is no grid-side converter,
and the rotor speed is a parameter, held by the turbine. Each of those is an entry in
docs/ASSUMPTIONS.md and is removed in a later step.

State vector (8):
    0, 1  psi_sD, psi_sQ   stator flux including the external series inductance
    2, 3  psi_rD, psi_rQ   rotor flux
    4, 5  vc_D, vc_Q       series capacitor voltage
    6, 7  x_D, x_Q         integrators of the rotor current controller, pu * s

With the speed held the model is linear, so its eigenvalues do not depend on the
operating point. See docs/EQUATIONS.md section 14.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .induction import IGData, InductionGenerator, _t

N_STATE = 8
STATE_NAMES = ("psi_sD", "psi_sQ", "psi_rD", "psi_rQ", "vc_D", "vc_Q", "x_D", "x_Q")
_MACHINE = ("r_e", "x_e", "r_s", "x_ls", "x_m", "r_r", "x_lr")
_ZERO = ("irref_D", "irref_Q", "vff_D", "vff_Q")


def gains_from_bandwidth(data: IGData, f_bw_hz: float) -> tuple[float, float]:
    """Kp (pu) and Ki (pu/s) that give the rotor current loop a bandwidth of f_bw_hz.

    Standard pole-cancellation tuning on the rotor transient inductance:
    Kp = alpha * sigma * L_r and Ki = alpha * R_r, with alpha in rad/s.
    """
    alpha = 2 * np.pi * f_bw_hz
    x_rr = data.x_lr + data.x_m
    sigma_x = x_rr - data.x_m**2 / (data.x_ls + data.x_m)
    return alpha * sigma_x / data.wb, alpha * data.r_r


@dataclass
class DFIG(InductionGenerator):
    data: IGData = field(default_factory=IGData)

    # ------------------------------------------------------------------ parameters
    def params(self, x_c, w_r=1.0, kp=0.0, ki=0.0, v_b=1.0, **overrides) -> dict:
        """Parameter dictionary for `rhs`. Every entry may carry leading batch dimensions.

        Keyword overrides replace any of r_e, x_e, r_s, x_ls, x_m, r_r, x_lr, and set the
        current references (irref_D, irref_Q) and the feed-forward voltage (vff_D, vff_Q).
        """
        base = {key: getattr(self.data, key) for key in _MACHINE}
        base.update(dict.fromkeys(_ZERO, 0.0))
        unknown = set(overrides) - set(base)
        if unknown:
            raise KeyError(f"unknown parameters: {sorted(unknown)}")
        base.update({k: v for k, v in overrides.items() if v is not None})
        p = {k: _t(v) for k, v in base.items()}
        p.update({"x_c": _t(x_c), "w_r": _t(w_r), "kp": _t(kp), "ki": _t(ki), "v_b": _t(v_b)})
        return p

    def rotor_voltage(self, x: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Converter output voltage (D, Q), referred to the stator."""
        c = self.currents(x, p)
        v_d = p["vff_D"] + p["kp"] * (p["irref_D"] - c["ir_D"]) + p["ki"] * x[..., 6]
        v_q = p["vff_Q"] + p["kp"] * (p["irref_Q"] - c["ir_Q"]) + p["ki"] * x[..., 7]
        return v_d, v_q

    # ------------------------------------------------------------------ dynamics
    def rhs(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        """dx/dt in seconds for state x of shape (..., 8)."""
        wb = self.data.wb
        c = self.currents(x, p)
        i_d, i_q, ir_d, ir_q = c["i_D"], c["i_Q"], c["ir_D"], c["ir_Q"]
        ps_d, ps_q, pr_d, pr_q = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        vc_d, vc_q = x[..., 4], x[..., 5]
        r = p["r_s"] + p["r_e"]
        slip = 1.0 - p["w_r"]
        vr_d, vr_q = self.rotor_voltage(x, p)

        dps_d = wb * (p["v_b"] + vc_d + r * i_d + ps_q)
        dps_q = wb * (vc_q + r * i_q - ps_d)
        dpr_d = wb * (vr_d - p["r_r"] * ir_d + slip * pr_q)
        dpr_q = wb * (vr_q - p["r_r"] * ir_q - slip * pr_d)
        dvc_d = wb * (p["x_c"] * i_d + vc_q)
        dvc_q = wb * (p["x_c"] * i_q - vc_d)
        dx_d = p["irref_D"] - ir_d
        dx_q = p["irref_Q"] - ir_q
        return torch.stack([dps_d, dps_q, dpr_d, dpr_q, dvc_d, dvc_q, dx_d, dx_q], dim=-1)

    # ------------------------------------------------------------------ steady state
    def equilibrium(
        self,
        x_c: float,
        w_r: float = 1.0,
        kp: float = 0.0,
        ki: float = 0.0,
        p_gen: float = 0.9,
        q_gen: float = 0.0,
        v_b: float = 1.0,
        **overrides,
    ) -> tuple[torch.Tensor, dict]:
        """Steady state delivering p_gen + j q_gen from the stator to the infinite bus.

        The current reference is set to the rotor current this needs. The steady rotor
        voltage is carried by the integrators, or by the feed-forward term when Ki = 0.
        """
        p = self.params(x_c, w_r=w_r, kp=kp, ki=ki, v_b=v_b, **overrides)
        q = {k: float(v) for k, v in p.items()}
        i = (p_gen - 1j * q_gen) / v_b
        v_c = -1j * x_c * i
        psi_s = -1j * (v_b + v_c + (q["r_s"] + q["r_e"]) * i)
        x_ss = q["x_ls"] + q["x_m"] + q["x_e"]
        i_r = (psi_s + x_ss * i) / q["x_m"]
        psi_r = -q["x_m"] * i + (q["x_lr"] + q["x_m"]) * i_r
        v_r = q["r_r"] * i_r + 1j * (1.0 - w_r) * psi_r

        integ = v_r / ki if ki > 0 else 0.0 + 0.0j
        v_ff = 0.0 + 0.0j if ki > 0 else v_r
        p.update(
            {
                "irref_D": _t(i_r.real),
                "irref_Q": _t(i_r.imag),
                "vff_D": _t(v_ff.real),
                "vff_Q": _t(v_ff.imag),
            }
        )
        x = np.array(
            [psi_s.real, psi_s.imag, psi_r.real, psi_r.imag, v_c.real, v_c.imag,
             integ.real, integ.imag]
        )  # fmt: skip
        return torch.as_tensor(x, dtype=torch.float64), p
