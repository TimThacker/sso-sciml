"""System B2, current-loop model, written a second time in phase coordinates.

An independent implementation for validation. It shares the parameter values with
`dfig.py` and nothing else:

    - quantities are phase a, b, c values in the stationary frame, not D and Q,
    - plain NumPy, no automatic differentiation,
    - integration by the trapezoidal rule with a linear solve per step, as an
      electromagnetic-transient program does, not explicit RK4.

State vector (11):
    0..2   stator flux per phase, including the line inductance
    3..5   rotor flux per phase, referred to the stator and to the stationary frame
    6..8   series capacitor voltage per phase
    9, 10  integrators of the rotor current controller, in the synchronous frame

The controller measures rotor current, turns it into the synchronous frame with the
angle of the infinite bus, and turns its output voltage back. That makes the system
linear with time-varying coefficients. See docs/EQUATIONS.md section 18.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_STATE = 11
_PHASE = np.array([0.0, 2 * np.pi / 3, -2 * np.pi / 3])
# Phase-coordinate form of a 90 degree rotation of the space vector.
_J = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]) / np.sqrt(3.0)


def to_phase(theta: float) -> np.ndarray:
    """3 x 2 matrix taking (D, Q) to phase values at frame angle theta."""
    return np.stack([np.cos(theta - _PHASE), -np.sin(theta - _PHASE)], axis=1)


def to_dq(theta: float) -> np.ndarray:
    """2 x 3 matrix taking phase values to (D, Q)."""
    return (2.0 / 3.0) * to_phase(theta).T


@dataclass(frozen=True)
class PhaseModel:
    """Parameters as plain floats. `wb` in rad/s, everything else per unit."""

    wb: float
    r_e: float
    x_e: float
    r_s: float
    x_ls: float
    x_m: float
    r_r: float
    x_lr: float
    x_c: float
    w_r: float
    kp: float
    ki: float
    v_b: float
    iref: tuple[float, float]
    vff: tuple[float, float]

    @classmethod
    def from_params(cls, p: dict, wb: float, **changes) -> PhaseModel:
        """Build from the parameter dictionary of `dfig.DFIG`."""
        q = {k: float(v) for k, v in p.items()}
        q.update(changes)
        keys = ("r_e", "x_e", "r_s", "x_ls", "x_m", "r_r", "x_lr", "x_c", "w_r", "kp", "ki", "v_b")
        return cls(
            wb=wb,
            iref=(q["irref_D"], q["irref_Q"]),
            vff=(q["vff_D"], q["vff_Q"]),
            **{k: q[k] for k in keys},
        )

    # ------------------------------------------------------------------ algebra
    def _current_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Stator and rotor phase currents as 3 x 11 maps of the state."""
        x_ss = self.x_ls + self.x_m + self.x_e
        x_rr = self.x_lr + self.x_m
        det = self.x_m**2 - x_ss * x_rr
        eye, zero = np.eye(3), np.zeros((3, 3))
        pad = np.zeros((3, 2))
        c_s = np.hstack([x_rr / det * eye, -self.x_m / det * eye, zero, pad])
        c_r = np.hstack([self.x_m / det * eye, -x_ss / det * eye, zero, pad])
        return c_s, c_r

    def system(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """A(t) and b(t) of dx/dt = A x + b, time in seconds."""
        theta = self.wb * t
        c_s, c_r = self._current_maps()
        t_p, t_dq = to_phase(theta), to_dq(theta)
        sel_psi_r = np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 5))])
        sel_vc = np.hstack([np.zeros((3, 6)), np.eye(3), np.zeros((3, 2))])
        sel_int = np.hstack([np.zeros((2, 9)), np.eye(2)])
        ir_dq = t_dq @ c_r  # rotor current in the controller frame
        v_r = t_p @ (-self.kp * ir_dq + self.ki * sel_int)  # state-dependent part
        a = np.vstack(
            [
                self.wb * ((self.r_s + self.r_e) * c_s + sel_vc),
                self.wb * (v_r - self.r_r * c_r + self.w_r * _J @ sel_psi_r),
                self.wb * self.x_c * c_s,
                -ir_dq,
            ]
        )
        iref, vff = np.array(self.iref), np.array(self.vff)
        b = np.concatenate(
            [
                self.wb * self.v_b * np.cos(theta - _PHASE),
                self.wb * (t_p @ (vff + self.kp * iref)),
                np.zeros(3),
                iref,
            ]
        )
        return a, b

    # ------------------------------------------------------------------ frames
    @staticmethod
    def from_dq_state(x_dq: np.ndarray, theta: float = 0.0) -> np.ndarray:
        """State of `dfig.DFIG` (8) to phase coordinates (11)."""
        t_p = to_phase(theta)
        return np.concatenate([t_p @ x_dq[0:2], t_p @ x_dq[2:4], t_p @ x_dq[4:6], x_dq[6:8]])

    def to_dq_state(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Trajectory in phase coordinates (n, 11) back to the state of `dfig.DFIG` (n, 8)."""
        out = np.empty((x.shape[0], 8))
        for n, tn in enumerate(np.atleast_1d(t)):
            t_dq = to_dq(self.wb * tn)
            out[n] = np.concatenate(
                [t_dq @ x[n, 0:3], t_dq @ x[n, 3:6], t_dq @ x[n, 6:9], x[n, 9:]]
            )
        return out

    def stator_current(self, x: np.ndarray) -> np.ndarray:
        """Phase currents out of the machine, shape (..., 3)."""
        return x @ self._current_maps()[0].T


def trapezoidal(model: PhaseModel, x0: np.ndarray, t_end: float, dt: float, save_every: int = 1):
    """Trapezoidal rule for dx/dt = A(t) x + b(t). Returns (t, x) with x of shape (n, 11)."""
    n_steps = int(round(t_end / dt))
    eye = np.eye(x0.size)
    x = x0.copy()
    a0, b0 = model.system(0.0)
    out, times = [x.copy()], [0.0]
    for k in range(1, n_steps + 1):
        a1, b1 = model.system(k * dt)
        x = np.linalg.solve(eye - 0.5 * dt * a1, (eye + 0.5 * dt * a0) @ x + 0.5 * dt * (b0 + b1))
        a0, b0 = a1, b1
        if k % save_every == 0:
            out.append(x.copy())
            times.append(k * dt)
    return np.array(times), np.array(out)
