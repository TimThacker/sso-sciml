"""Frequency-domain view of a converter on a network: admittance, impedance and the
screens built on them.

The converter is described by its 2 x 2 admittance in the synchronous frame, Y(s): the
current it draws for a small change of its terminal voltage. That is what a frequency
scan of a vendor's black-box model returns. The network is described by its impedance
seen from the same terminal, Z(s). The two meet in

    det(I + Y(s) Z(s)) = 0

whose roots are the modes of the whole system. Frequencies are in the synchronous frame:
0 Hz here is 60 Hz in the phases, and -40 Hz here is 20 Hz in the phases.
"""

from __future__ import annotations

import numpy as np
import torch

from .physics.converter import WB, Converter, active_states


def scan_grid(f_min: float = 0.01, f_max: float = 2000.0, n: int = 2000) -> np.ndarray:
    """Frequencies (Hz, synchronous frame) from -f_max to f_max, dense near zero."""
    pos = np.geomspace(f_min, f_max, n)
    return np.concatenate([-pos[::-1], [0.0], pos])


def network_impedance(net: dict, s: np.ndarray) -> np.ndarray:
    """Z(s) of the network seen from the converter terminal, shape (len(s), 2, 2).

    s in 1/s, synchronous frame. Transformer in series with a compensated line in
    parallel with an uncompensated path. Keys: r_e, x_e, x_c, r_p, x_p, r_t, x_t.
    """
    s = np.asarray(s, dtype=complex)

    def z(sig):
        line = net["r_e"] + sig * net["x_e"] + net["x_c"] / sig
        path = net["r_p"] + sig * net["x_p"]
        return net["r_t"] + sig * net["x_t"] + line * path / (line + path)

    plus, minus = z(s / WB + 1j), z(s / WB - 1j)
    z_d, z_q = 0.5 * (plus + minus), (plus - minus) / 2j
    return np.stack([np.stack([z_d, -z_q], -1), np.stack([z_q, z_d], -1)], -2)


class ConverterScan:
    """Small-signal model of a converter alone at one operating point."""

    def __init__(self, model: Converter, v_t: complex, i: complex, p: dict):
        z0 = model.terminal_equilibrium(v_t, i)
        v0 = torch.tensor([v_t.real, v_t.imag], dtype=torch.float64)
        a = torch.func.jacrev(lambda z: model.rhs_terminal(z, v0, p))(z0).numpy()
        b = torch.func.jacrev(lambda v: model.rhs_terminal(z0, v, p))(v0).numpy()
        keep = active_states(model.controls.follow, n=a.shape[0])
        self.a, self.b = a[np.ix_(keep, keep)], b[keep]
        self.poles, vec = np.linalg.eig(self.a)
        # Y(s) = -C (sI - A)^-1 B with C picking the current: a sum over the poles.
        self._left = -vec[:2]
        self._right = np.linalg.solve(vec, self.b)

    @property
    def unstable_poles(self) -> int:
        """Poles of the converter alone, on an ideal voltage, with positive real part."""
        return int((self.poles.real > 1e-9).sum())

    def admittance(self, s: np.ndarray) -> np.ndarray:
        """Y(s), shape (len(s), 2, 2): current into the converter per terminal voltage."""
        s = np.asarray(s, dtype=complex)
        gain = 1.0 / (s[:, None] - self.poles[None, :])
        return np.einsum("ik,sk,kj->sij", self._left, gain, self._right)


def positive_sequence(m: np.ndarray) -> np.ndarray:
    """The part of a 2 x 2 synchronous-frame matrix that maps a rotating vector to one
    rotating the same way. The coupling to the mirror frequency is left out. This is what
    a single-frequency positive-sequence scan measures."""
    return 0.5 * (m[..., 0, 0] + m[..., 1, 1]) + 0.5j * (m[..., 1, 0] - m[..., 0, 1])


def passivity_index(y: np.ndarray) -> np.ndarray:
    """Smallest eigenvalue of the Hermitian part of Y. Negative where the converter can
    feed an oscillation."""
    return np.linalg.eigvalsh(0.5 * (y + np.conj(np.swapaxes(y, -1, -2))))[..., 0]


def unstable_modes(y: np.ndarray, z: np.ndarray, open_loop: int = 0, coupled: bool = True) -> int:
    """Number of unstable modes of converter and network together, by the generalised
    Nyquist count from the scans alone. y and z are taken on `scan_grid`, in rising
    order of frequency. `open_loop` is the number of unstable poles the two have apart.
    With `coupled` false only the positive-sequence parts of the scans are used.
    """
    if coupled:
        det = np.linalg.det(np.eye(2) + y @ z)
    else:
        det = 1.0 + positive_sequence(y) * positive_sequence(z)
    turns = np.diff(np.unwrap(np.angle(det))).sum() / (2 * np.pi)
    return open_loop - int(np.round(turns))


def small_gain_score(y: np.ndarray, z: np.ndarray) -> float:
    """Largest gain of the loop Y Z over the frequencies at which Y is not passive.

    Below 1 the system is stable, whatever else is true of it: where the converter is
    passive a passive network cannot be destabilised, and where it is not the loop gain
    is too small. Zero if the converter is passive at every frequency.
    """
    bad = passivity_index(y) < 0
    if not bad.any():
        return 0.0
    return float(np.linalg.norm(y[bad] @ z[bad], ord=2, axis=(-2, -1)).max())


def scan_growth(y: np.ndarray, z: np.ndarray, f_hz: np.ndarray) -> tuple[float, float]:
    """Growth rate (1/s) and frequency (Hz) from positive-sequence scans alone.

    At each frequency where the reactance of converter and network in series passes from
    capacitive to inductive, the growth rate is -R / (dX/dw). The largest is returned. If
    the reactance has no such crossing the result is (-inf, nan).
    """
    total = positive_sequence(z) + 1.0 / positive_sequence(y)
    w = 2 * np.pi * f_hz
    x, r = total.imag, total.real
    up = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if up.size == 0:
        return float("-inf"), float("nan")
    frac = -x[up] / (x[up + 1] - x[up])
    slope = (x[up + 1] - x[up]) / (w[up + 1] - w[up])
    growth = -(r[up] + frac * (r[up + 1] - r[up])) / slope
    top = int(np.argmax(growth))
    return float(growth[top]), float(
        f_hz[up[top]] + frac[top] * (f_hz[up[top] + 1] - f_hz[up[top]])
    )
