"""System B2 with full controls, written a second time in phase coordinates.

A check on `dfig_full.py` (docs/EQUATIONS.md section 15) in the way `dfig_abc.py`
checks the current-loop model. It takes the parameter values of `dfig_full.DFIGFull`
and shares nothing else with it:

    - electrical quantities are phase a, b, c values in the stationary frame;
    - the states are the machine's own fluxes, the grid-side filter current and the
      capacitor voltage, not loop fluxes; the terminal voltage is solved for at each
      evaluation from the three branches that meet at the terminal;
    - the phase-locked loop carries the absolute angle of its frame, not the angle
      relative to the infinite bus;
    - plain NumPy, no automatic differentiation; integrated by an adaptive
      Runge-Kutta method of SciPy, not fixed-step RK4.

State vector (27):
     0..2   stator flux of the machine, per phase
     3..5   rotor flux, referred to the stator and to the stationary frame
     6..8   grid-side filter current, converter to terminal
     9..11  series capacitor voltage
    12      rotor speed, pu
    13      angle of the PLL frame, absolute, rad
    14      PLL integrator
    15, 16  rotor current controller integrators (PLL frame)
    17, 18  filtered active and reactive power at the terminal
    19, 20  outer loop integrators
    21, 22  grid-side current controller integrators (PLL frame)
    23      DC link voltage
    24      DC voltage controller integrator
    25, 26  filtered terminal voltage, PLL frame

Conventions: stator current `i` out of the machine, rotor current `i_r` into the rotor,
line current `i_L = i + i_g` from the terminal towards the infinite bus.

    stator     v_t = (1/wb) dpsi_s/dt - R_s i,      psi_s = -X_s i + X_m i_r
    rotor      (1/wb) dpsi_r/dt = v_r - R_r i_r + w_r J psi_r,   psi_r = -X_m i + X_r i_r
    filter     (X_f/wb) di_g/dt = v_g - v_t - R_f i_g
    line       (X_E/wb) di_L/dt = v_t - v_b - v_c - R_E i_L
    capacitor  (1/wb) dv_c/dt = X_C i_L
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

N_STATE = 27
_PHASE = np.array([0.0, 2 * np.pi / 3, -2 * np.pi / 3])
_J = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]) / np.sqrt(3.0)


def to_phase(angle: float) -> np.ndarray:
    """3 x 2: (d, q) in a frame at `angle` to phase values."""
    return np.stack([np.cos(angle - _PHASE), -np.sin(angle - _PHASE)], axis=1)


def to_dq(angle: float) -> np.ndarray:
    """2 x 3: phase values to (d, q) in a frame at `angle`."""
    return (2.0 / 3.0) * to_phase(angle).T


class PhaseFull:
    """`p` is a parameter dictionary of `dfig_full.DFIGFull`, read as plain floats."""

    def __init__(self, p: dict, wb: float):
        q = {k: float(v) for k, v in p.items()}
        self.q, self.wb = q, wb
        self.x_s, self.x_r = q["x_ls"] + q["x_m"], q["x_lr"] + q["x_m"]
        self.det = q["x_m"] ** 2 - self.x_s * self.x_r

    def machine_currents(self, psi_s: np.ndarray, psi_r: np.ndarray):
        q, det = self.q, self.det
        i = (self.x_r * psi_s - q["x_m"] * psi_r) / det
        i_r = (q["x_m"] * psi_s - self.x_s * psi_r) / det
        return i, i_r

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        q, wb = self.q, self.wb
        psi_s, psi_r, i_g, v_c = x[0:3], x[3:6], x[6:9], x[9:12]
        w_r, ang, x_pll = x[12], x[13], x[14]
        i, i_r = self.machine_currents(psi_s, psi_r)
        i_l = i + i_g
        v_b = q["v_b"] * np.cos(wb * t - _PHASE)
        dq = to_dq(ang)

        # Rotor-side converter, in the PLL frame. Its voltage is known before the
        # terminal voltage, since it depends on states only.
        p_ref = q["k_opt"] * w_r**3
        ird_ref = q["out_kp"] * (p_ref - x[17]) + q["out_ki"] * x[19]
        irq_ref = -(q["out_kp"] * (q["q_ref"] - x[18]) + q["out_ki"] * x[20])
        irc = dq @ i_r
        vr_c = q["kp"] * (np.array([ird_ref, irq_ref]) - irc) + q["ki"] * x[15:17]
        v_r = to_phase(ang) @ vr_c

        # Grid-side converter.
        igd_ref = -(q["dc_kp"] * (q["vdc_ref"] - x[23]) + q["dc_ki"] * x[24])
        igc = dq @ i_g
        vg_c = q["ff"] * x[25:27] + q["g_kp"] * (np.array([igd_ref, 0.0]) - igc) \
            + q["g_ki"] * x[21:23]  # fmt: skip
        v_g = to_phase(ang) @ vg_c

        # Terminal voltage, phase by phase, from di_L/dt = di/dt + di_g/dt.
        #   dpsi_s/dt = wb (v_t + R_s i)
        #   dpsi_r/dt = wb (v_r - R_r i_r + w_r J psi_r)
        #   di/dt     = (X_r dpsi_s/dt - X_m dpsi_r/dt) / det
        #   di_g/dt   = wb (v_g - v_t - R_f i_g) / X_f
        #   di_L/dt   = wb (v_t - v_b - v_c - R_E i_L) / X_E
        dpsi_r = wb * (v_r - q["r_r"] * i_r + w_r * _J @ psi_r)
        c_s = wb * self.x_r / self.det  # di/dt per unit of (v_t + R_s i)
        known = c_s * q["r_s"] * i - q["x_m"] * dpsi_r / self.det \
            + wb * (v_g - q["r_f"] * i_g) / q["x_f"]  # fmt: skip
        gain = c_s - wb / q["x_f"]  # coefficient of v_t in di_L/dt from the branches
        # wb (v_t - v_b - v_c - R_E i_L) / X_E = gain v_t + known
        v_t = (known + wb * (v_b + v_c + q["r_e"] * i_l) / q["x_e"]) / (wb / q["x_e"] - gain)
        dpsi_s = wb * (v_t + q["r_s"] * i)
        di_g = wb * (v_g - v_t - q["r_f"] * i_g) / q["x_f"]
        dv_c = wb * q["x_c"] * i_l

        # Measurements in the PLL frame.
        vtc = dq @ v_t
        ilc = dq @ i_l
        p_t = vtc[0] * ilc[0] + vtc[1] * ilc[1]
        q_t = vtc[1] * ilc[0] - vtc[0] * ilc[1]
        # Torque from the stator flux and current in a stationary frame.
        s0 = to_dq(0.0)
        ps, cs = s0 @ psi_s, s0 @ i
        t_e = ps[0] * cs[1] - ps[1] * cs[0]
        p_rsc = -(vr_c[0] * irc[0] + vr_c[1] * irc[1])
        p_gsc = vg_c[0] * igc[0] + vg_c[1] * igc[1]

        out = np.empty(N_STATE)
        out[0:3], out[3:6], out[6:9], out[9:12] = dpsi_s, dpsi_r, di_g, dv_c
        out[12] = (q["p_m"] / w_r - t_e) / (2.0 * q["h"])
        out[13] = wb + q["pll_kp"] * vtc[1] + q["pll_ki"] * x_pll
        out[14] = vtc[1]
        out[15:17] = np.array([ird_ref, irq_ref]) - irc
        out[17] = (p_t - x[17]) / q["t_f"]
        out[18] = (q_t - x[18]) / q["t_f"]
        out[19] = p_ref - x[17]
        out[20] = q["q_ref"] - x[18]
        out[21:23] = np.array([igd_ref, 0.0]) - igc
        out[23] = (p_rsc - p_gsc) / (q["c_dc"] * x[23])
        out[24] = q["vdc_ref"] - x[23]
        out[25:27] = (vtc - x[25:27]) / q["t_ff"]
        return out

    # ------------------------------------------------------------------ frames
    def from_full_state(self, x23: np.ndarray) -> np.ndarray:
        """A state of `dfig_full.DFIGFull` (synchronous frame, angle 0 at t = 0) in this
        model's coordinates. Electrical quantities are recomputed from that model's
        loop fluxes through the flux equations written out here."""
        q = self.q
        x_e, x_f = q["x_e"], q["x_f"]
        a = x_e / (x_f + x_e)
        x_ss = self.x_s + a * x_f
        det = q["x_m"] ** 2 - x_ss * self.x_r
        out = np.zeros(N_STATE)
        sync = to_phase(0.0)
        cur = {}
        for ax, (i1, ir, i2) in {"D": (0, 2, 6), "Q": (1, 3, 7)}.items():
            psi = x23[i1] + a * x23[i2]
            i_s = (self.x_r * psi - q["x_m"] * x23[ir]) / det
            i_g = (x23[i2] - x_e * i_s) / (x_f + x_e)
            cur[ax] = (i_s, i_g)
        i_s = np.array([cur["D"][0], cur["Q"][0]])
        i_g = np.array([cur["D"][1], cur["Q"][1]])
        psi_r = x23[2:4]
        i_r = (psi_r + q["x_m"] * i_s) / self.x_r
        psi_s = -self.x_s * i_s + q["x_m"] * i_r
        out[0:3], out[3:6] = sync @ psi_s, sync @ psi_r
        out[6:9], out[9:12] = sync @ i_g, sync @ x23[4:6]
        out[12], out[13] = x23[8], x23[9]
        out[14:27] = x23[10:23]
        return out

    def capacitor_dq(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Capacitor voltage in the synchronous frame, (n, 2)."""
        return np.array([to_dq(self.wb * tn) @ x[n, 9:12] for n, tn in enumerate(t)])


def simulate(model: PhaseFull, x0: np.ndarray, t_eval: np.ndarray, rtol: float = 1e-10,
             atol: float = 1e-12) -> np.ndarray:  # fmt: skip
    """Adaptive Dormand-Prince integration. Returns (n_t, 27)."""
    sol = solve_ivp(model.rhs, (float(t_eval[0]), float(t_eval[-1])), x0, method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)  # fmt: skip
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T
