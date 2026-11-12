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
from .physics.dfig_full import DFIGFull


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


class Scan:
    """A plant alone on an ideal voltage as dz/dt = A z + B v, current drawn = C z.

    Y(s) = C (sI - A)^-1 B is evaluated as a sum over the poles.
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        self.a, self.b, self.c = a, b, c
        self.poles, vec = np.linalg.eig(a)
        self._left = c @ vec
        self._right = np.linalg.solve(vec, b)

    @property
    def unstable_poles(self) -> int:
        """Poles of the converter alone, on an ideal voltage, with positive real part."""
        return int((self.poles.real > 1e-9).sum())

    def admittance(self, s: np.ndarray) -> np.ndarray:
        """Y(s), shape (len(s), 2, 2): current into the converter per terminal voltage."""
        s = np.asarray(s, dtype=complex)
        gain = 1.0 / (s[:, None] - self.poles[None, :])
        return np.einsum("ik,sk,kj->sij", self._left, gain, self._right)


class ConverterScan(Scan):
    """Small-signal model of a converter alone at one operating point."""

    def __init__(self, model: Converter, v_t: complex, i: complex, p: dict):
        z0 = model.terminal_equilibrium(v_t, i)
        v0 = torch.tensor([v_t.real, v_t.imag], dtype=torch.float64)
        a = torch.func.jacrev(lambda z: model.rhs_terminal(z, v0, p))(z0).numpy()
        b = torch.func.jacrev(lambda v: model.rhs_terminal(z0, v, p))(v0).numpy()
        keep = active_states(model.controls.follow, n=a.shape[0])
        # The first two states are the current into the converter.
        super().__init__(a[np.ix_(keep, keep)], b[keep], -np.eye(len(keep))[:2])


class DFIGScan(Scan):
    """Small-signal model of the doubly-fed plant alone (System B2, full controls) at the
    operating point of a case on any network.

    The plant is cut from its network at the terminals. `dfig_full.py` with a line of
    zero impedance is the plant on an ideal voltage, and there the capacitor voltage
    enters the equations exactly where a change of terminal voltage would. So the
    columns of the state matrix that belong to the capacitor voltage are B, and those
    two states are dropped.
    """

    def __init__(self, model: DFIGFull, x0: torch.Tensor, p: dict):
        plant = DFIGFull(data=model.data, controls=model.controls)
        c = model.currents(x0, p)
        vt = model.terminal_voltage(x0, p)
        x_s = p["x_ls"] + p["x_m"]
        z0 = x0[:23].clone()
        for n, ax in enumerate("DQ"):
            z0[n] = -x_s * c[f"i_{ax}"] + p["x_m"] * c[f"ir_{ax}"]  # stator flux
            z0[6 + n] = p["x_f"] * c[f"ig_{ax}"]
            z0[4 + n] = vt[n] - (p["v_b"] if n == 0 else 0.0)
        zero = torch.zeros((), dtype=torch.float64)
        q = {**p, "r_e": zero, "x_e": zero, "x_c": zero}
        keep = [k for k in range(23) if k not in (4, 5)]
        self.residual = float(plant.rhs(z0, q)[keep].abs().max())

        def current(z):
            cur = plant.currents(z, q)
            return torch.stack([cur["il_D"], cur["il_Q"]])

        a = torch.func.jacrev(lambda z: plant.rhs(z, q))(z0).numpy()
        out = torch.func.jacrev(current)(z0).numpy()
        # il is the current out of the plant; the admittance is of the current drawn.
        super().__init__(a[np.ix_(keep, keep)], a[np.ix_(keep, [4, 5])], -out[:, keep])


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


def loop_scores(y: np.ndarray, z: np.ndarray, guard_deg: tuple[float, ...] = ()) -> dict:
    """Scores built on the eigenvalues of the loop matrix Y Z at each frequency.

    The system is stable if the plant alone and the network alone are stable and, as the
    loop is closed gradually (Y Z scaled from 0 to 1), no eigenvalue of the scaled loop
    matrix ever equals -1 at any frequency. That holds wherever

        norm    the largest singular value of Y Z is below 1 (step 17), or
        radius  the largest eigenvalue of Y Z is below 1 in size, which is weaker, or
        the plant is passive, because the network is.

    `norm` and `radius` are the largest values over the frequencies at which the plant
    is not passive; each is zero if it is passive everywhere. A score below 1 proves
    stability.

    `guard_<angle>` is the largest size of any eigenvalue that lies within the given
    angle of the negative real axis, at any frequency. Below 1 says the eigenvalues pass
    -1 on the inside or stay clear of it by that angle: a gain margin with a guard
    sector. With a guard of zero and a continuous scan this is exact. With a finite
    guard it needs no passivity and no count of turns.
    """
    lam = np.linalg.eigvals(y @ z)
    size = np.abs(lam)
    bad = passivity_index(y) < 0
    out = {
        "norm": float(np.linalg.norm(y[bad] @ z[bad], ord=2, axis=(-2, -1)).max())
        if bad.any()
        else 0.0,
        "radius": float(size[bad].max()) if bad.any() else 0.0,
    }
    off_axis = np.abs(np.angle(-lam, deg=True))
    for guard in guard_deg:
        near = off_axis < guard
        out[f"guard_{guard:g}"] = float(size[near].max()) if near.any() else 0.0
    return out


GUARDS = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0)  # guard sectors, degrees
STEPS = (4, 16, 40, 80)  # keep every n-th point of `scan_grid`: 1,001, 251, 101, 51 points
NOISE = (0.05, 0.20)  # relative error added to each entry of Y, on the 251-point scan
NOISE_STEP = 16
SCREEN_COLUMNS = (
    "count", "count_pos", "norm", "radius",
    *(f"guard_{g:g}" for g in GUARDS),
    "pos_guard_10", "pos_guard_30",
    *(f"{name}_s{k}" for k in STEPS for name in ("count", "guard_10", "guard_30", "guard_60")),
    *(f"{name}_n{round(100 * e)}" for e in NOISE for name in ("count", "guard_30", "guard_60")),
)  # fmt: skip


def screen_scores(y: np.ndarray, z: np.ndarray, open_loop: int, seed: int) -> np.ndarray:
    """Every scan-based score of one case, in the order of SCREEN_COLUMNS.

    y and z are taken on `scan_grid()`. The coarse scans keep every n-th point. The
    noisy scans multiply each entry of Y at each frequency by 1 + e, with e complex
    normal of the given standard deviation, drawn from `seed`.
    """
    full = loop_scores(y, z, GUARDS)
    pos = (positive_sequence(y) * positive_sequence(z))[:, None, None]
    one = np.ones_like(pos)
    pos_guard = loop_scores(pos, one, (10.0, 30.0))
    out = [
        unstable_modes(y, z, open_loop),
        unstable_modes(y, z, open_loop, coupled=False),
        full["norm"], full["radius"],
        *(full[f"guard_{g:g}"] for g in GUARDS),
        pos_guard["guard_10"], pos_guard["guard_30"],
    ]  # fmt: skip
    for k in STEPS:
        coarse = loop_scores(y[::k], z[::k], (10.0, 30.0, 60.0))
        out += [unstable_modes(y[::k], z[::k], open_loop),
                coarse["guard_10"], coarse["guard_30"], coarse["guard_60"]]  # fmt: skip
    rng = np.random.default_rng(seed)
    y_c, z_c = y[::NOISE_STEP], z[::NOISE_STEP]
    for level in NOISE:
        err = rng.normal(size=(*y_c.shape, 2)) @ np.array([1.0, 1j]) * level / np.sqrt(2)
        noisy = loop_scores(y_c * (1 + err), z_c, (30.0, 60.0))
        out += [unstable_modes(y_c * (1 + err), z_c, open_loop),
                noisy["guard_30"], noisy["guard_60"]]  # fmt: skip
    return np.array(out, dtype=float)


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
