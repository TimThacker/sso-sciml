"""System B2, second stage: doubly-fed machine with its full control structure.

Adds to `dfig.py`:
    - a phase-locked loop on the terminal voltage; all controllers work in its frame,
    - outer loops on filtered active and reactive power that set the rotor current
      references, with the active power reference following rotor speed cubed,
    - a grid-side converter behind a filter reactor, with current control,
    - a DC link whose voltage the grid-side converter regulates,
    - rotor speed as a state, driven by constant mechanical power.

State vector (23):
     0, 1   lam1_D, lam1_Q   flux of the stator-plus-line loop
     2, 3   psi_rD, psi_rQ   rotor flux
     4, 5   vc_D, vc_Q       series capacitor voltage
     6, 7   lam2_D, lam2_Q   flux of the grid-side-filter-plus-line loop
     8      w_r              rotor speed, pu
     9      theta            PLL angle relative to the infinite bus, rad
    10      x_pll            PLL integrator
    11, 12  x_rd, x_rq       rotor current controller integrators (PLL frame)
    13, 14  p_f, q_f         filtered active and reactive power at the terminal
    15, 16  x_p, x_q         outer loop integrators
    17, 18  x_gd, x_gq       grid-side current controller integrators (PLL frame)
    19      v_dc             DC link voltage, pu
    20      x_dc             DC voltage controller integrator
    21, 22  v_fd, v_fq       filtered terminal voltage (PLL frame), fed forward to the
                             grid-side converter voltage

See docs/EQUATIONS.md section 15 for the derivation and sign conventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from scipy.optimize import brentq, fsolve

from .dfig import gains_from_bandwidth
from .induction import IGData, _t

N_STATE = 23
W_R, THETA = 8, 9
STATE_NAMES = (
    "lam1_D", "lam1_Q", "psi_rD", "psi_rQ", "vc_D", "vc_Q", "lam2_D", "lam2_Q", "w_r", "theta",
    "x_pll", "x_rd", "x_rq", "p_f", "q_f", "x_p", "x_q", "x_gd", "x_gq", "v_dc", "x_dc",
    "v_fd", "v_fq",
)  # fmt: skip
_MACHINE = ("r_e", "x_e", "r_s", "x_ls", "x_m", "r_r", "x_lr", "h")


@dataclass(frozen=True)
class Controls:
    """Controller settings, given as bandwidths where a standard tuning rule exists."""

    rsc_hz: float = 10.0  # rotor current loop bandwidth
    pll_hz: float = 10.0  # PLL natural frequency
    pll_zeta: float = 0.7
    outer_hz: float = 1.0  # power loop crossover
    t_f: float = 0.02  # power measurement filter, s
    x_f: float = 0.30  # grid-side filter reactance, pu on the plant base
    r_f: float = 0.003
    gsc_hz: float = 100.0  # grid-side current loop bandwidth
    c_dc: float = 0.02  # DC capacitance, pu * s (stored energy 0.01 pu * s at 1 pu)
    dc_hz: float = 10.0  # DC voltage loop natural frequency
    dc_zeta: float = 0.7
    w_rated: float = 1.2  # rotor speed at which the speed-cubed curve gives 1 pu power
    ff: float = 1.0  # terminal-voltage feed-forward gain of the grid-side converter (0 or 1)
    t_ff: float = 0.002  # feed-forward filter, s
    # Rotor-side converter voltage limit, pu. The default is far above any command and so
    # means no limit. It is finite because an infinite value gives 0 * inf = NaN in the
    # automatic derivative of the limiter.
    vr_max: float = 1e6

    def gains(self, data: IGData) -> dict:
        kp, ki = gains_from_bandwidth(data, self.rsc_hz)
        wn_pll = 2 * np.pi * self.pll_hz
        wn_dc = 2 * np.pi * self.dc_hz
        w_out = 2 * np.pi * self.outer_hz
        a_g = 2 * np.pi * self.gsc_hz
        return {
            "kp": kp,
            "ki": ki,
            "pll_kp": 2 * self.pll_zeta * wn_pll,
            "pll_ki": wn_pll**2,
            "out_kp": w_out * self.t_f,
            "out_ki": w_out,
            "t_f": self.t_f,
            "x_f": self.x_f,
            "r_f": self.r_f,
            "g_kp": a_g * self.x_f / data.wb,
            "g_ki": a_g * self.r_f,
            "c_dc": self.c_dc,
            "dc_kp": 2 * self.dc_zeta * wn_dc * self.c_dc,
            "dc_ki": wn_dc**2 * self.c_dc,
            "k_opt": 1.0 / self.w_rated**3,
            "ff": self.ff,
            "t_ff": self.t_ff,
            "vr_max": self.vr_max,
        }


def _rot(d, q, cos, sin):
    """PLL frame to synchronous frame."""
    return d * cos - q * sin, d * sin + q * cos


def _unrot(big_d, big_q, cos, sin):
    """Synchronous frame to PLL frame."""
    return big_d * cos + big_q * sin, -big_d * sin + big_q * cos


@dataclass
class DFIGFull:
    data: IGData = field(default_factory=IGData)
    controls: Controls = field(default_factory=Controls)

    # ------------------------------------------------------------------ parameters
    def params(self, x_c, v_b=1.0, p_m=0.0, q_ref=0.0, vdc_ref=1.0, **overrides) -> dict:
        """Parameter dictionary for `rhs`. Every entry may carry leading batch dimensions.

        Keyword overrides replace any machine, network or controller value by name.
        """
        base = {key: getattr(self.data, key) for key in _MACHINE}
        base.update(self.controls.gains(self.data))
        unknown = set(overrides) - set(base)
        if unknown:
            raise KeyError(f"unknown parameters: {sorted(unknown)}")
        base.update({k: v for k, v in overrides.items() if v is not None})
        base.update({"x_c": x_c, "v_b": v_b, "p_m": p_m, "q_ref": q_ref, "vdc_ref": vdc_ref})
        return {k: _t(v) for k, v in base.items()}

    # ------------------------------------------------------------------ algebra
    def currents(self, x: torch.Tensor, p: dict) -> dict:
        """Stator (out of the machine), rotor, grid-side converter and line currents."""
        a = p["x_e"] / (p["x_f"] + p["x_e"])
        x_ss = p["x_ls"] + p["x_m"] + a * p["x_f"]
        x_rr = p["x_lr"] + p["x_m"]
        x_m = p["x_m"]
        det = x_m * x_m - x_ss * x_rr
        out = {}
        for ax, (i1, ir, i2) in {"D": (0, 2, 6), "Q": (1, 3, 7)}.items():
            psi = x[..., i1] + a * x[..., i2]
            i_s = (x_rr * psi - x_m * x[..., ir]) / det
            i_g = (x[..., i2] - p["x_e"] * i_s) / (p["x_f"] + p["x_e"])
            out[f"i_{ax}"] = i_s
            out[f"ir_{ax}"] = (x_m * psi - x_ss * x[..., ir]) / det
            out[f"ig_{ax}"] = i_g
            out[f"il_{ax}"] = i_s + i_g
        return out

    def _network(self, x: torch.Tensor, p: dict, c: dict, vg_d, vg_q) -> dict:
        """Derivatives of the network fluxes and capacitor voltage, and the stator flux.

        This and the two methods around it are the only parts that know the network
        topology. Here: a single series line.
        """
        wb = self.data.wb
        i_d, i_q, ig_d, ig_q = c["i_D"], c["i_Q"], c["ig_D"], c["ig_Q"]
        il_d, il_q = c["il_D"], c["il_Q"]
        l1_d, l1_q, vc_d, vc_q = x[..., 0], x[..., 1], x[..., 4], x[..., 5]
        l2_d, l2_q = x[..., 6], x[..., 7]
        return {
            "dl1": (
                wb * (p["v_b"] + vc_d + p["r_s"] * i_d + p["r_e"] * il_d + l1_q),
                wb * (vc_q + p["r_s"] * i_q + p["r_e"] * il_q - l1_d),
            ),
            "dvc": (wb * (p["x_c"] * il_d + vc_q), wb * (p["x_c"] * il_q - vc_d)),
            "dl2": (
                wb * (vg_d - p["v_b"] - vc_d - p["r_f"] * ig_d - p["r_e"] * il_d + l2_q),
                wb * (vg_q - vc_q - p["r_f"] * ig_q - p["r_e"] * il_q - l2_d),
            ),
            "psi_s": (l1_d + p["x_e"] * il_d, l1_q + p["x_e"] * il_q),
            "extra": [],
        }

    def _gsc_current_rate(self, p: dict, net: dict, dpr_d, dpr_q) -> tuple:
        """Time derivative of the grid-side converter current, from the flux derivatives."""
        a = p["x_e"] / (p["x_f"] + p["x_e"])
        x_ss = p["x_ls"] + p["x_m"] + a * p["x_f"]
        x_rr = p["x_lr"] + p["x_m"]
        det = p["x_m"] * p["x_m"] - x_ss * x_rr
        out = []
        for dl1, dl2, dpr in zip(net["dl1"], net["dl2"], (dpr_d, dpr_q), strict=True):
            di = (x_rr * (dl1 + a * dl2) - p["x_m"] * dpr) / det
            out.append((dl2 - p["x_e"] * di) / (p["x_f"] + p["x_e"]))
        return out[0], out[1]

    def _evaluate(self, x: torch.Tensor, p: dict) -> dict:
        """Everything `rhs` needs, also used for outputs."""
        wb = self.data.wb
        c = self.currents(x, p)
        i_d, i_q, ir_d, ir_q = c["i_D"], c["i_Q"], c["ir_D"], c["ir_Q"]
        ig_d, ig_q, il_d, il_q = c["ig_D"], c["ig_Q"], c["il_D"], c["il_Q"]
        pr_d, pr_q = x[..., 2], x[..., 3]
        w_r, theta = x[..., W_R], x[..., THETA]
        cos, sin = torch.cos(theta), torch.sin(theta)

        # Rotor-side converter: outer power loops, then current loop, in the PLL frame.
        p_ref = p["k_opt"] * w_r**3
        ird_ref = p["out_kp"] * (p_ref - x[..., 13]) + p["out_ki"] * x[..., 15]
        irq_ref = -(p["out_kp"] * (p["q_ref"] - x[..., 14]) + p["out_ki"] * x[..., 16])
        irc_d, irc_q = _unrot(ir_d, ir_q, cos, sin)
        # Controller command, then the converter's voltage limit: the command keeps its
        # direction and its magnitude is held at vr_max. Below the limit the factor is
        # exactly one.
        cmd_d = p["kp"] * (ird_ref - irc_d) + p["ki"] * x[..., 11]
        cmd_q = p["kp"] * (irq_ref - irc_q) + p["ki"] * x[..., 12]
        size = torch.sqrt(cmd_d * cmd_d + cmd_q * cmd_q)
        keep = torch.clamp(p["vr_max"] / torch.clamp(size, min=1e-12), max=1.0)
        vr_d, vr_q = _rot(keep * cmd_d, keep * cmd_q, cos, sin)

        # Grid-side converter: DC voltage loop sets the d-axis current, q-axis reference zero.
        igd_ref = -(p["dc_kp"] * (p["vdc_ref"] - x[..., 19]) + p["dc_ki"] * x[..., 20])
        igc_d, igc_q = _unrot(ig_d, ig_q, cos, sin)
        vg_d, vg_q = _rot(
            p["ff"] * x[..., 21] + p["g_kp"] * (igd_ref - igc_d) + p["g_ki"] * x[..., 17],
            p["ff"] * x[..., 22] + p["g_kp"] * (0.0 - igc_q) + p["g_ki"] * x[..., 18],
            cos,
            sin,
        )

        slip = 1.0 - w_r
        dpr_d = wb * (vr_d - p["r_r"] * ir_d + slip * pr_q)
        dpr_q = wb * (vr_q - p["r_r"] * ir_q - slip * pr_d)
        net = self._network(x, p, c, vg_d, vg_q)
        (dl1_d, dl1_q), (dvc_d, dvc_q), (dl2_d, dl2_q) = net["dl1"], net["dvc"], net["dl2"]

        # Terminal voltage from the grid-side branch; it needs the current derivatives.
        dig_d, dig_q = self._gsc_current_rate(p, net, dpr_d, dpr_q)
        vt_d = vg_d - p["r_f"] * ig_d - p["x_f"] * (dig_d / wb - ig_q)
        vt_q = vg_q - p["r_f"] * ig_q - p["x_f"] * (dig_q / wb + ig_d)
        vtc_d, vtc_q = _unrot(vt_d, vt_q, cos, sin)

        p_t = vt_d * il_d + vt_q * il_q
        q_t = vt_q * il_d - vt_d * il_q
        ps_d, ps_q = net["psi_s"]  # the machine's own stator flux
        t_e = ps_d * i_q - ps_q * i_d
        p_rsc = -(vr_d * ir_d + vr_q * ir_q)  # from the rotor into the DC link
        p_gsc = vg_d * ig_d + vg_q * ig_q  # from the DC link into the grid

        deriv = torch.stack(
            [
                dl1_d, dl1_q, dpr_d, dpr_q, dvc_d, dvc_q, dl2_d, dl2_q,
                (p["p_m"] / w_r - t_e) / (2.0 * p["h"]),
                p["pll_kp"] * vtc_q + p["pll_ki"] * x[..., 10],
                vtc_q,
                ird_ref - irc_d,
                irq_ref - irc_q,
                (p_t - x[..., 13]) / p["t_f"],
                (q_t - x[..., 14]) / p["t_f"],
                p_ref - x[..., 13],
                p["q_ref"] - x[..., 14],
                igd_ref - igc_d,
                0.0 - igc_q,
                (p_rsc - p_gsc) / (p["c_dc"] * x[..., 19]),
                p["vdc_ref"] - x[..., 19],
                (vtc_d - x[..., 21]) / p["t_ff"],
                (vtc_q - x[..., 22]) / p["t_ff"],
                *net["extra"],
            ],
            dim=-1,
        )  # fmt: skip
        return {"deriv": deriv, "vt": (vt_d, vt_q), "vr": (vr_d, vr_q), "vg": (vg_d, vg_q),
                "vr_command": (cmd_d, cmd_q),
                "p_t": p_t, "q_t": q_t, "t_e": t_e, "p_rsc": p_rsc, "p_gsc": p_gsc}  # fmt: skip

    def rhs(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        """dx/dt in seconds for state x of shape (..., 23)."""
        return self._evaluate(x, p)["deriv"]

    def outputs(self, x: torch.Tensor, p: dict) -> dict:
        """Terminal voltage, converter voltages, powers and torque at a state."""
        out = self._evaluate(x, p)
        out.pop("deriv")
        return out

    def terminal_voltage(self, x: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self._evaluate(x, p)["vt"]

    # ------------------------------------------------------------------ steady state
    def equilibrium(
        self,
        x_c: float,
        w_r: float = 1.0,
        q_ref: float = 0.0,
        v_b: float = 1.0,
        **overrides,
    ) -> tuple[torch.Tensor, dict]:
        """Steady state at a rotor speed. Power follows the speed-cubed curve.

        The mechanical power that holds the speed is solved for and returned in the
        parameter dictionary.
        """
        p = self.params(x_c, v_b=v_b, q_ref=q_ref, **overrides)
        q = {k: float(v) for k, v in p.items()}
        p_total = q["k_opt"] * w_r**3
        z_line = self._network_impedance(q, x_c)  # seen from the terminals at 60 Hz

        def mismatch(u):
            i_l = complex(u[0], u[1])
            s = (v_b + z_line * i_l) * np.conj(i_l)
            return [s.real - p_total, s.imag - q_ref]

        sol = fsolve(mismatch, [p_total / v_b, 0.0], xtol=1e-12)
        i_l = complex(sol[0], sol[1])
        v_t = v_b + z_line * i_l
        theta = np.angle(v_t)
        rot = np.exp(1j * theta)
        x_s, x_rr = q["x_ls"] + q["x_m"], q["x_lr"] + q["x_m"]

        def split(igd: float) -> dict:
            i_g = igd * rot
            i = i_l - i_g
            psi_s = -1j * (v_t + q["r_s"] * i)
            i_r = (psi_s + x_s * i) / q["x_m"]
            psi_r = -q["x_m"] * i + x_rr * i_r
            v_r = q["r_r"] * i_r + 1j * (1.0 - w_r) * psi_r
            v_g = v_t + (q["r_f"] + 1j * q["x_f"]) * i_g
            return {"i_g": i_g, "i": i, "psi_s": psi_s, "i_r": i_r, "psi_r": psi_r, "v_r": v_r,
                    "v_g": v_g,
                    "dc": -(v_r * np.conj(i_r)).real - (v_g * np.conj(i_g)).real}  # fmt: skip

        igd = brentq(lambda g: split(g)["dc"], -1.0, 1.0, xtol=1e-15)
        s = split(igd)
        lam1, lam2, v_c, extra = self._network_steady(q, x_c, i_l, s["psi_s"], s["i_g"])
        irc, vrc = s["i_r"] / rot, s["v_r"] / rot
        vgc = s["v_g"] / rot - q["ff"] * abs(v_t)  # the feed-forward carries the rest
        t_e = (np.conj(s["psi_s"]) * s["i"]).imag

        x = np.array(
            [
                lam1.real, lam1.imag, s["psi_r"].real, s["psi_r"].imag, v_c.real, v_c.imag,
                lam2.real, lam2.imag, w_r, theta, 0.0,
                vrc.real / q["ki"], vrc.imag / q["ki"],
                p_total, q_ref,
                irc.real / q["out_ki"], -irc.imag / q["out_ki"],
                vgc.real / q["g_ki"], vgc.imag / q["g_ki"],
                1.0, -igd / q["dc_ki"],
                abs(v_t), 0.0,
                *extra,
            ]
        )  # fmt: skip
        p["p_m"] = _t(t_e * w_r)
        return torch.as_tensor(x, dtype=torch.float64), p

    def _network_impedance(self, q: dict, x_c: float) -> complex:
        return q["r_e"] + 1j * (q["x_e"] - x_c)

    def _network_steady(self, q: dict, x_c: float, i_l: complex, psi_s: complex, i_g: complex):
        """Loop fluxes, capacitor voltage and any extra network states in steady state."""
        lam1 = psi_s - q["x_e"] * i_l
        lam2 = q["x_f"] * i_g + q["x_e"] * i_l
        return lam1, lam2, -1j * x_c * i_l, []

    def describe(self) -> dict:
        """Controller gains in use, for logging."""
        return {**asdict(self.controls), **self.controls.gains(self.data)}
