"""What an instrument sees of a response, and the modes read back from it (task T3).

The dataset holds the line current as a deviation in the synchronous frame. A fault
recorder sees one phase current: the fundamental with the deviation riding on it,
sampled at 32 points per cycle. A phasor measurement unit sees less: one phasor per
frame, each the one-cycle Fourier sum of that phase current.

A mode at f in the synchronous frame shows in the phase current at 60 - f and 60 + f.
In the phasor it shows at f again, but a unit reporting at 60 frames per second cannot
tell f from 60 - f, and at 30 frames per second not from 30 - f either.

Modes are read back with the matrix pencil method or with Prony's method.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .data import t2

F0 = 60.0
PER_CYCLE = 32
RECORDER_DT = 1.0 / (F0 * PER_CYCLE)


def recorder_times() -> np.ndarray:
    return np.arange(int(np.floor(t2.T_END / RECORDER_DT)) + 1) * RECORDER_DT


def phase_current(observed: np.ndarray, fundamental: float = 1.0) -> np.ndarray:
    """Phase current at the recorder's sampling instants, pu: the fundamental plus the
    deviation of the line current. `observed` is (cases, N_T, 5) as in the dataset."""
    t = recorder_times()
    dev = CubicSpline(t2.times(), observed[:, :, 0] + 1j * observed[:, :, 1], axis=1)(t)
    return ((fundamental + dev) * np.exp(2j * np.pi * F0 * t)).real


def add_noise(y: np.ndarray, signal: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """White noise at `snr_db` below the root-mean-square size of each row of `signal`."""
    if not np.isfinite(snr_db):
        return y.copy()
    size = np.sqrt((np.abs(signal) ** 2).mean(axis=1, keepdims=True))
    return y + size * 10 ** (-snr_db / 20) * rng.standard_normal(y.shape)


def phasors(current: np.ndarray, frames_per_s: float) -> tuple[np.ndarray, float]:
    """One-cycle Fourier phasor of each row at the reporting instants, with the steady
    phasor taken off, and the time between frames. The first frame ends one cycle in."""
    step = int(round(1.0 / (frames_per_s * RECORDER_DT)))
    k = np.arange(PER_CYCLE)
    ends = np.arange(PER_CYCLE, current.shape[1] + 1, step)
    out = np.empty((current.shape[0], len(ends)), dtype=complex)
    for j, e in enumerate(ends):
        t = (e - PER_CYCLE + k) * RECORDER_DT
        out[:, j] = 2.0 / PER_CYCLE * (current[:, e - PER_CYCLE : e]
                                       * np.exp(-2j * np.pi * F0 * t)).sum(axis=1)  # fmt: skip
    return out, step * RECORDER_DT


def matrix_pencil(y: np.ndarray, dt: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Complex frequencies s (1/s) and amplitudes of `order` exponentials fitted to one
    signal sampled every dt."""
    n = len(y)
    pencil = n // 3
    rows = np.array([y[i : i + pencil + 1] for i in range(n - pencil)])
    _, sv, vh = np.linalg.svd(rows, full_matrices=False)
    order = min(order, int((sv > 1e-9 * sv[0]).sum()))  # a clean signal may hold fewer
    v = vh[:order].T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    return _amplitudes(y, z, dt)


def prony(y: np.ndarray, dt: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """As `matrix_pencil`, by least-squares linear prediction and the roots of its
    polynomial."""
    n = len(y)
    rows = np.array([y[i : i + order][::-1] for i in range(n - order)])
    a = np.linalg.lstsq(rows, y[order:], rcond=None)[0]
    z = np.roots(np.concatenate([[1.0], -a]))
    return _amplitudes(y, z[np.abs(z) > 0], dt)


def _amplitudes(y: np.ndarray, z: np.ndarray, dt: float):
    basis = z[None, :] ** np.arange(len(y))[:, None]
    size = np.abs(basis).max(axis=0)  # a fast-growing column would swamp the others
    amp = np.linalg.lstsq(basis / size, y, rcond=None)[0] / size
    return np.log(z) / dt, amp


def strongest(s: np.ndarray, amp: np.ndarray, n: int, dt: float, band: tuple) -> complex:
    """The fitted exponential with the most energy over the window among those whose
    frequency (Hz, signed) lies in the band. NaN if there is none."""
    f = s.imag / (2 * np.pi)
    ok = (f > band[0]) & (f < band[1]) & (np.abs(s.real) < 200.0)
    if not ok.any():
        return complex(np.nan, np.nan)
    t = np.arange(n) * dt
    energy = (np.abs(amp[ok, None] * np.exp(s.real[ok, None] * t)) ** 2).sum(axis=1)
    return complex(s[ok][int(np.argmax(energy))])


def from_recorder(current: np.ndarray, order: int = 12, method=matrix_pencil) -> np.ndarray:
    """Growth rate (1/s) and synchronous-frame frequency (Hz) of the sub-synchronous
    mode from each phase current: (cases, 2)."""
    out = np.empty((current.shape[0], 2))
    for i, y in enumerate(current):
        s, amp = method(y, RECORDER_DT, order)
        m = strongest(s, amp, len(y), RECORDER_DT, (5.0, 55.0))
        out[i] = m.real, F0 - m.imag / (2 * np.pi)
    return out


def from_phasors(ph: np.ndarray, dt: float, order: int = 6) -> np.ndarray:
    """Growth rate and apparent frequency (Hz, magnitude) of the strongest oscillation
    of each phasor record: (cases, 2). The frequency is what the frame rate lets
    through, not folded back."""
    out = np.empty((ph.shape[0], 2))
    limit = 0.5 / dt
    for i, y in enumerate(ph):
        s, amp = matrix_pencil(y, dt, min(order, len(y) // 3))
        f = s.imag / (2 * np.pi)
        amp = np.where(np.abs(f) > 2.0, amp, 0.0)  # leave out what does not oscillate
        m = strongest(s, amp, len(y), dt, (-limit - 1.0, limit + 1.0))
        out[i] = m.real, abs(m.imag) / (2 * np.pi)
    return out


def folded(f: np.ndarray, frames_per_s: float) -> np.ndarray:
    """The frequency at which a mode at f appears at a given frame rate."""
    r = np.mod(f, frames_per_s)
    return np.minimum(r, frames_per_s - r)
