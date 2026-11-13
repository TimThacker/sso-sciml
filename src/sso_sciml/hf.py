"""Control delay and a filter capacitor in the generic converter (lifts part of A30).

Two things the converter of step 17 leaves out, both of which act above the band of the
earlier steps:

    delay       the converter voltage follows its command after a time t_d (sampling
                and modulation). Per axis in the synchronous frame, as a second-order
                Pade approximant: two states per axis. The rotation of the frame during
                the delay is taken as compensated.
    capacitor   a shunt branch at the terminal, between the filter reactor and the
                transformer: susceptance b_c in series with a damping resistor r_d. With
                the reactor and the network it forms the LCL filter.

Both are added to the converter alone, as it is scanned. The modes of the whole are those
of converter and shunt branch side by side, closed on the network exactly
(`vectorfit.closed_loop_modes`). There is no 14-state network model with these in it.

Used by scripts/delay_and_capacitor.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.optimize import fsolve

from . import admittance as adm
from . import vectorfit as vf
from .data import c1
from .data.t1 import case_seed
from .physics.converter import WB, Converter

J = vf.J
BANDS = {"slow": (0.0, 2.0), "sub": (2.0, 58.0), "mid": (58.0, 2000.0), "high": (2000.0, np.inf)}
VARIANTS = ("base", "delay", "capacitor", "both")
RANGES = {  # sampled evenly on a logarithmic scale
    "rho": (0.02, 0.10),  # delay times current-loop bandwidth
    "b_c": (0.02, 0.10),  # filter capacitor susceptance, pu
    "r_d": (0.02, 0.50),  # damping resistor in series with it, pu
}
COLUMNS = (
    "t_d", "b_c", "r_d", "f_lcl",
    *(f"{name}_{v}" for v in VARIANTS
      for name in ("solved", "top", "f_top", *BANDS, "alone")),
    "fit_2k", "fit_10k", "f_passive", "passive_above_58",
)  # fmt: skip
_GRID_2K = adm.scan_grid()[:: adm.NOISE_STEP]
_GRID_10K = adm.scan_grid(f_max=10000.0)[:: adm.NOISE_STEP]
_GRID_FINE = adm.scan_grid(f_max=10000.0)[::4]


@dataclass
class DelayedConverter(Converter):
    """The converter alone with a delay between voltage command and voltage."""

    t_d: float = 0.0

    def rhs_terminal(self, z: torch.Tensor, v: torch.Tensor, p: dict) -> torch.Tensor:
        """z (..., 14): current, the 8 control states, then two delay states per axis."""
        i_d, i_q, c, w = z[..., 0], z[..., 1], z[..., 2:10], z[..., 10:]
        vt_d, vt_q = v[..., 0], v[..., 1]
        cmd = self._command(c, i_d, i_q, p)
        t = self.t_d
        # H(s) = 1 - (12 s / T) / (s^2 + 6 s / T + 12 / T^2), per axis.
        vg = [cmd["vg"][k] - 12.0 / t * w[..., 2 * k + 1] for k in range(2)]
        lag = []
        for k in range(2):
            x1, x2 = w[..., 2 * k], w[..., 2 * k + 1]
            lag += [x2, cmd["vg"][k] - 12.0 / t**2 * x1 - 6.0 / t * x2]
        return torch.stack(
            [
                WB * (vg[0] - vt_d - p["r_f"] * i_d + p["x_f"] * i_q) / p["x_f"],
                WB * (vg[1] - vt_q - p["r_f"] * i_q - p["x_f"] * i_d) / p["x_f"],
                *self._control_rates(c, p, {**cmd, "vg": tuple(vg)}, i_d, i_q, vt_d, vt_q),
                *lag,
            ],
            dim=-1,
        )

    def terminal_equilibrium(self, v_t: complex, i: complex) -> torch.Tensor:
        base = super().terminal_equilibrium(v_t, i)
        v_g = v_t + (self.controls.r_f + 1j * self.controls.x_f) * i
        scale = self.t_d**2 / 12.0
        delay = torch.tensor([v_g.real * scale, 0.0, v_g.imag * scale, 0.0], dtype=torch.float64)
        return torch.cat([base, delay])


class PlantScan(adm.Scan):
    """Converter, with or without delay, beside a shunt capacitor branch, as one plant:
    dx/dt = A x + B v, current drawn = C x + D v."""

    def __init__(self, model: Converter, v_t: complex, i: complex, p: dict,
                 b_c: float = 0.0, r_d: float = 0.0):  # fmt: skip
        z0 = model.terminal_equilibrium(v_t, i)
        v0 = torch.tensor([v_t.real, v_t.imag], dtype=torch.float64)
        a = torch.func.jacrev(lambda z: model.rhs_terminal(z, v0, p))(z0).numpy()
        b = torch.func.jacrev(lambda v: model.rhs_terminal(z0, v, p))(v0).numpy()
        keep = [k for k in range(a.shape[0]) if not (model.controls.follow and k in (2, 3))]
        a, b = a[np.ix_(keep, keep)], b[keep]
        c = -np.eye(len(keep))[:2]
        self.d = np.zeros((2, 2))
        if b_c > 0:
            # Capacitor voltage v_c behind the resistor: the branch draws (v - v_c) / r_d.
            a = block_diag(a, WB * (-np.eye(2) / (r_d * b_c) - J))
            b = np.concatenate([b, WB * np.eye(2) / (r_d * b_c)])
            c = np.concatenate([c, -np.eye(2) / r_d], axis=1)
            self.d = np.eye(2) / r_d
        super().__init__(a, b, c)

    def admittance(self, s: np.ndarray) -> np.ndarray:
        return super().admittance(s) + self.d


def _z60(net: dict) -> complex:
    line = net["r_e"] + 1j * (net["x_e"] - net["x_c"])
    path = net["r_p"] + 1j * net["x_p"]
    return net["r_t"] + 1j * net["x_t"] + line * path / (line + path)


def operating_point(model: Converter, p_src: float, net: dict, b_c: float, r_d: float):
    """Terminal voltage, converter current and reactive current reference with the shunt
    branch at the terminal. Raises ValueError without a steady state."""
    y_sh = 1.0 / (r_d - 1j / b_c) if b_c > 0 else 0.0
    z_g = _z60(net)
    follow = bool(model.controls.follow)

    def unpack(w):
        if follow:
            v = complex(w[0], w[1])
            return v, model.current_at(v, p_src), 0.0
        v = np.exp(1j * w[0])
        return v, model.current_at(v, p_src, float(w[1])), float(w[1])

    def mismatch(w):
        v, i, _ = unpack(w)
        r = v - 1.0 - z_g * (i - y_sh * v)
        return [r.real, r.imag]

    with np.errstate(all="ignore"):
        w = fsolve(mismatch, [1.0, 0.0] if follow else [0.0, 0.0], xtol=1e-12,
                   full_output=True)[0]  # fmt: skip
        ok = np.all(np.isfinite(w)) and np.max(np.abs(mismatch(w))) < 1e-9
    if not ok:
        raise ValueError("steady state not found")
    return unpack(w)


def banded(lam: np.ndarray) -> list[float]:
    """Largest real part (1/s) of the modes in each of BANDS; -inf for an empty band."""
    f = np.abs(lam.imag) / (2 * np.pi)
    return [float(lam.real[(f >= lo) & (f < hi)].max()) if ((f >= lo) & (f < hi)).any()
            else float("-inf") for lo, hi in BANDS.values()]  # fmt: skip


def solve(theta: np.ndarray, follow: float, t_d: float, b_c: float, r_d: float):
    """Plant and network of one case with a delay and a capacitor (either may be zero)."""
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    net = c1.network(th)
    controls = c1.controls(th, follow)
    model = DelayedConverter(controls, t_d=t_d) if t_d > 0 else Converter(controls)
    v, i, iq = operating_point(model, th["p"], net, b_c, r_d)
    return PlantScan(model, v, i, model.params(th["p"], iq_ref=iq), b_c, r_d), net


def lcl_frequency(theta: np.ndarray, b_c: float) -> float:
    """Resonance of the capacitor with the filter reactor and the 60 Hz reactance of the
    network in parallel, Hz in the phases."""
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    x_grid, x_f = abs(_z60(c1.network(th)).imag), th["x_f"]
    return float(60.0 * np.sqrt((x_f + x_grid) / (x_f * x_grid * b_c)))


def case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of COLUMNS: the converter of dataset C1 as it is, with a
    delay, with a filter capacitor, and with both."""
    kind, theta = args
    follow = float(kind == "follow")
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    rng = np.random.default_rng(case_seed(theta) + 28)
    rho, b_c, r_d = (float(np.exp(rng.uniform(*np.log(RANGES[k])))) for k in RANGES)
    t_d = rho / th["cc_hz"]
    out = [t_d, b_c, r_d, lcl_frequency(theta, b_c)]
    both = None
    for variant in VARIANTS:
        delay = t_d if variant in ("delay", "both") else 0.0
        cap = b_c if variant in ("capacitor", "both") else 0.0
        try:
            plant, net = solve(theta, follow, delay, cap, r_d)
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            out += [0.0] + [np.nan] * (3 + len(BANDS))
            continue
        lam = vf.closed_loop_modes(plant.a, plant.b, plant.c, plant.d, net)
        top = int(np.argmax(lam.real))
        out += [1.0, lam.real[top], abs(lam.imag[top]) / (2 * np.pi), *banded(lam),
                plant.poles.real.max()]  # fmt: skip
        if variant == "both":
            both = (plant, net)
    if both is None:
        return np.array([*out, np.nan, np.nan, np.nan, np.nan])
    plant, net = both
    fits = []
    for grid in (_GRID_2K, _GRID_10K):
        s = 2j * np.pi * grid
        fits.append(adm.fit_growth(plant.admittance(s), adm.network_impedance(net, s), grid)[0])
    bad = adm.passivity_index(plant.admittance(2j * np.pi * _GRID_FINE)) < 0
    freq = np.abs(_GRID_FINE)
    return np.array([
        *out, *fits,
        float(freq[bad].max()) if bad.any() else 0.0,
        float((bad & (freq > 58.0)).any()),
    ])  # fmt: skip


def _init_worker() -> None:
    torch.set_num_threads(1)


def cases(kind: str, theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """`case` for every row of theta. Row order is kept."""
    jobs = [(kind, row) for row in theta]
    if processes <= 1:
        return np.array([case(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(case, jobs, chunksize=32))
