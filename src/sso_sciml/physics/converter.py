"""System C: a generic voltage-source converter on a series-compensated or weak network.

No machine and no slip. One model covers the converters that meet a network through a
filter reactor and a current loop:

    - power out (`p_src` > 0): full-converter wind or solar plant, storage discharging,
    - power in (`p_src` < 0): active rectifier of a drive, a UPS, a power supply, storage
      charging,
    - synchronised by a phase-locked loop (`follow` = 0), or with a current reference that
      follows the measured voltage waveform and no phase-locked loop (`follow` = 1), as
      in a power-factor-correcting supply.

In every case the DC-link voltage loop sets the size of the current, so the converter
holds its power constant at frequencies inside that loop.

Network: converter, filter reactor, terminal, transformer, then a series-compensated line
in parallel with an uncompensated path, to an infinite bus.

State vector (14), synchronous frame of the infinite bus unless noted:
     0, 1   il_D, il_Q   current in the compensated line
     2, 3   ip_D, ip_Q   current in the parallel path
     4, 5   vc_D, vc_Q   series capacitor voltage
     6      theta        PLL angle, rad (held at zero when follow = 1)
     7      x_pll        PLL integrator
     8, 9   x_cd, x_cq   current controller integrators (control frame)
    10      v_dc         DC-link voltage, pu
    11      x_dc         DC voltage controller integrator
    12, 13  v_fd, v_fq   filtered terminal voltage (control frame)

Current is positive out of the converter. `rhs_terminal` is the same converter alone,
driven by its terminal voltage: 10 states, the converter current followed by states 6 to
13. Its small-signal admittance is what `sso_sciml.admittance` works with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import fsolve

from .induction import _t

WB = 2 * np.pi * 60.0
N_STATE = 14
N_TERMINAL = 10
THETA, X_PLL, V_DC = 6, 7, 10
STATE_NAMES = (
    "il_D", "il_Q", "ip_D", "ip_Q", "vc_D", "vc_Q", "theta", "x_pll", "x_cd", "x_cq", "v_dc",
    "x_dc", "v_fd", "v_fq",
)  # fmt: skip


@dataclass(frozen=True)
class ConverterControls:
    cc_hz: float = 200.0  # current loop bandwidth
    pll_hz: float = 10.0  # PLL natural frequency
    pll_zeta: float = 0.7
    dc_hz: float = 10.0  # DC voltage loop natural frequency
    dc_zeta: float = 0.7
    c_dc: float = 0.02  # DC capacitance, pu * s
    ff: float = 1.0  # terminal-voltage feed-forward gain
    t_ff: float = 0.002  # voltage measurement filter, s
    x_f: float = 0.15  # filter reactance, pu
    r_f: float = 0.003
    follow: float = 0.0  # 0: phase-locked loop. 1: current reference follows the voltage

    def gains(self) -> dict:
        a = 2 * np.pi * self.cc_hz
        wn_pll, wn_dc = 2 * np.pi * self.pll_hz, 2 * np.pi * self.dc_hz
        return {
            "g_kp": a * self.x_f / WB,
            "g_ki": a * self.r_f,
            "pll_kp": 2 * self.pll_zeta * wn_pll,
            "pll_ki": wn_pll**2,
            "c_dc": self.c_dc,
            "dc_kp": 2 * self.dc_zeta * wn_dc * self.c_dc,
            "dc_ki": wn_dc**2 * self.c_dc,
            "ff": self.ff,
            "t_ff": self.t_ff,
            "x_f": self.x_f,
            "r_f": self.r_f,
            "follow": self.follow,
        }


def _rot(d, q, cos, sin):
    """Control frame to synchronous frame."""
    return d * cos - q * sin, d * sin + q * cos


def _unrot(big_d, big_q, cos, sin):
    """Synchronous frame to control frame."""
    return big_d * cos + big_q * sin, -big_d * sin + big_q * cos


def active_states(follow: float, n: int = N_STATE) -> list[int]:
    """States that move. Without a phase-locked loop its two states are held."""
    first = THETA if n == N_STATE else 2
    held = {first, first + 1} if follow else set()
    return [k for k in range(n) if k not in held]


@dataclass
class Converter:
    controls: ConverterControls = field(default_factory=ConverterControls)

    def params(self, p_src, iq_ref=0.0, v_b=1.0, vdc_ref=1.0, **network) -> dict:
        """Parameter dictionary. Network keys: r_e, x_e, x_c, r_p, x_p, r_t, x_t."""
        base = self.controls.gains()
        base.update(network)
        base.update({"p_src": p_src, "iq_ref": iq_ref, "v_b": v_b, "vdc_ref": vdc_ref})
        return {k: _t(v) for k, v in base.items()}

    # ------------------------------------------------------------------ controls
    def _command(self, c: torch.Tensor, i_d, i_q, p: dict) -> dict:
        """Converter voltage from the control states c (..., 8) and the current."""
        cos, sin = torch.cos(c[..., 0]), torch.sin(c[..., 0])
        f = p["follow"]
        u = -(p["dc_kp"] * (p["vdc_ref"] - c[..., 4]) + p["dc_ki"] * c[..., 5])
        vf_d, vf_q = c[..., 6], c[..., 7]
        ic_d, ic_q = _unrot(i_d, i_q, cos, sin)
        e_d = u * ((1.0 - f) + f * vf_d) - ic_d
        e_q = (1.0 - f) * p["iq_ref"] + f * u * vf_q - ic_q
        vg_d, vg_q = _rot(
            p["ff"] * vf_d + p["g_kp"] * e_d + p["g_ki"] * c[..., 2],
            p["ff"] * vf_q + p["g_kp"] * e_q + p["g_ki"] * c[..., 3],
            cos,
            sin,
        )
        return {"vg": (vg_d, vg_q), "e": (e_d, e_q), "cos": cos, "sin": sin}

    def _control_rates(self, c: torch.Tensor, p: dict, cmd: dict, i_d, i_q, vt_d, vt_q) -> list:
        f = p["follow"]
        vtc_d, vtc_q = _unrot(vt_d, vt_q, cmd["cos"], cmd["sin"])
        vg_d, vg_q = cmd["vg"]
        p_conv = vg_d * i_d + vg_q * i_q  # from the DC link towards the network
        return [
            (1.0 - f) * (p["pll_kp"] * vtc_q + p["pll_ki"] * c[..., 1]),
            (1.0 - f) * vtc_q,
            cmd["e"][0],
            cmd["e"][1],
            (p["p_src"] - p_conv) / (p["c_dc"] * c[..., 4]),
            p["vdc_ref"] - c[..., 4],
            # With follow = 1 the filter acts on the waveform, not on its envelope.
            (vtc_d - c[..., 6]) / p["t_ff"] + f * WB * c[..., 7],
            (vtc_q - c[..., 7]) / p["t_ff"] - f * WB * c[..., 6],
        ]

    # ------------------------------------------------------------------ dynamics
    def _evaluate(self, x: torch.Tensor, p: dict) -> dict:
        il_d, il_q, ip_d, ip_q, vc_d, vc_q = (x[..., k] for k in range(6))
        i_d, i_q = il_d + ip_d, il_q + ip_q
        c = x[..., 6:]
        cmd = self._command(c, i_d, i_q, p)
        vg_d, vg_q = cmd["vg"]
        x_s, r_s = p["x_f"] + p["x_t"], p["r_f"] + p["r_t"]
        # Voltage left for the two loop inductances: through the line, through the path.
        com_d = vg_d - p["v_b"] - r_s * i_d + x_s * i_q
        com_q = vg_q - r_s * i_q - x_s * i_d
        e1_d = com_d - p["r_e"] * il_d + p["x_e"] * il_q - vc_d
        e1_q = com_q - p["r_e"] * il_q - p["x_e"] * il_d - vc_q
        e2_d = com_d - p["r_p"] * ip_d + p["x_p"] * ip_q
        e2_q = com_q - p["r_p"] * ip_q - p["x_p"] * ip_d
        det = (x_s + p["x_e"]) * (x_s + p["x_p"]) - x_s * x_s
        dil_d = WB * ((x_s + p["x_p"]) * e1_d - x_s * e2_d) / det
        dil_q = WB * ((x_s + p["x_p"]) * e1_q - x_s * e2_q) / det
        dip_d = WB * ((x_s + p["x_e"]) * e2_d - x_s * e1_d) / det
        dip_q = WB * ((x_s + p["x_e"]) * e2_q - x_s * e1_q) / det
        vt_d = vg_d - p["r_f"] * i_d + p["x_f"] * i_q - p["x_f"] * (dil_d + dip_d) / WB
        vt_q = vg_q - p["r_f"] * i_q - p["x_f"] * i_d - p["x_f"] * (dil_q + dip_q) / WB
        deriv = torch.stack(
            [
                dil_d, dil_q, dip_d, dip_q,
                WB * (p["x_c"] * il_d + vc_q),
                WB * (p["x_c"] * il_q - vc_d),
                *self._control_rates(c, p, cmd, i_d, i_q, vt_d, vt_q),
            ],
            dim=-1,
        )  # fmt: skip
        return {"deriv": deriv, "vt": (vt_d, vt_q), "vg": (vg_d, vg_q), "i": (i_d, i_q)}

    def rhs(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        """dx/dt in seconds for state x of shape (..., 14)."""
        return self._evaluate(x, p)["deriv"]

    def terminal_voltage(self, x: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self._evaluate(x, p)["vt"]

    def rhs_terminal(self, z: torch.Tensor, v: torch.Tensor, p: dict) -> torch.Tensor:
        """The converter alone. z (..., 10): current then control states. v (..., 2):
        terminal voltage in the synchronous frame."""
        i_d, i_q, c = z[..., 0], z[..., 1], z[..., 2:]
        vt_d, vt_q = v[..., 0], v[..., 1]
        cmd = self._command(c, i_d, i_q, p)
        vg_d, vg_q = cmd["vg"]
        return torch.stack(
            [
                WB * (vg_d - vt_d - p["r_f"] * i_d + p["x_f"] * i_q) / p["x_f"],
                WB * (vg_q - vt_q - p["r_f"] * i_q - p["x_f"] * i_d) / p["x_f"],
                *self._control_rates(c, p, cmd, i_d, i_q, vt_d, vt_q),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------ steady state
    def _lag(self) -> complex:
        """Steady-state gain of the voltage measurement in the control frame."""
        c = self.controls
        return 1.0 / (1.0 + 1j * WB * c.t_ff) if c.follow else 1.0 + 0.0j

    def control_state(self, v_t: complex, i: complex) -> tuple[np.ndarray, float]:
        """Control states (8) that hold current i at terminal voltage v_t, and the
        reactive current reference that goes with them."""
        c = self.controls
        g = c.gains()
        theta = 0.0 if c.follow else float(np.angle(v_t))
        rot = np.exp(1j * theta)
        v_f = self._lag() * v_t / rot
        i_c = i / rot
        u = (i_c / v_f).real if c.follow else i_c.real
        iq_ref = 0.0 if c.follow else i_c.imag
        v_gc = (v_t + (c.r_f + 1j * c.x_f) * i) / rot - c.ff * v_f
        state = np.array(
            [theta, 0.0, v_gc.real / g["g_ki"], v_gc.imag / g["g_ki"], 1.0, -u / g["dc_ki"],
             v_f.real, v_f.imag]
        )  # fmt: skip
        return state, iq_ref

    def current_at(self, v_t: complex, p_src: float, iq_ref: float = 0.0) -> complex:
        """Current that carries p_src out of the DC link at terminal voltage v_t."""
        c = self.controls
        if c.follow:
            h = self._lag()
            # i = u h v_t ;  Re((v_t + z_f i) conj(i)) = p_src
            a, b = c.r_f * abs(h * v_t) ** 2, abs(v_t) ** 2 * h.real
        else:
            a, b = c.r_f, abs(v_t)
            p_src = p_src - c.r_f * iq_ref**2
        u = (-b + np.sqrt(b * b + 4 * a * p_src)) / (2 * a)
        if c.follow:
            return u * self._lag() * v_t
        return (u + 1j * iq_ref) * np.exp(1j * np.angle(v_t))

    def terminal_equilibrium(self, v_t: complex, i: complex) -> torch.Tensor:
        """Steady state of the converter alone, for `rhs_terminal`."""
        ctrl, _ = self.control_state(v_t, i)
        return torch.as_tensor(np.concatenate([[i.real, i.imag], ctrl]), dtype=torch.float64)

    def equilibrium(
        self, p_src: float, v_set: float = 1.0, v_b: float = 1.0, **network
    ) -> tuple[torch.Tensor, dict]:
        """Steady state on the network. With a phase-locked loop the reactive current
        holds the terminal voltage at v_set. Without one the current follows the measured
        voltage. Raises ValueError if there is no such operating point."""
        c = self.controls
        n = {k: float(v) for k, v in network.items()}
        z_line = n["r_e"] + 1j * (n["x_e"] - n["x_c"])
        z_path = n["r_p"] + 1j * n["x_p"]
        z_g = n["r_t"] + 1j * n["x_t"] + z_line * z_path / (z_line + z_path)
        z_f = c.r_f + 1j * c.x_f

        if c.follow:
            h = self._lag()

            def unpack(w):
                v_t = v_b / (1.0 - z_g * w[0] * h)
                return v_t, w[0] * h * v_t

            def mismatch(w):
                v_t, i = unpack(w)
                return [((v_t + z_f * i) * np.conj(i)).real - p_src]

            start = [p_src]
        else:

            def unpack(w):
                i = complex(w[0], w[1])
                return v_b + z_g * i, i

            def mismatch(w):
                v_t, i = unpack(w)
                return [((v_t + z_f * i) * np.conj(i)).real - p_src, abs(v_t) - v_set]

            start = [p_src, 0.0]

        # The solver's own flag is not used: at this tolerance it reports slow progress on
        # points that satisfy the equations to rounding.
        sol = fsolve(mismatch, start, xtol=1e-13, full_output=True)[0]
        if not np.isfinite(sol).all() or max(abs(m) for m in mismatch(sol)) > 1e-9:
            raise ValueError("steady state not found")
        v_t, i = unpack(sol)
        ctrl, iq_ref = self.control_state(v_t, i)
        i_l = i * z_path / (z_line + z_path)
        i_p = i - i_l
        v_c = -1j * n["x_c"] * i_l
        x = np.concatenate([[i_l.real, i_l.imag, i_p.real, i_p.imag, v_c.real, v_c.imag], ctrl])
        p = self.params(p_src, iq_ref=iq_ref, v_b=v_b, **network)
        return torch.as_tensor(x, dtype=torch.float64), p
