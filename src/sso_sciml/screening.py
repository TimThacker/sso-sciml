"""Deterministic screens for the sub-synchronous mode, ordered by the data they need.

    topology    network reactances and a generic machine leakage reactance
    slip        the above and the rotor speed
    resistance  the above, machine resistances and the current-loop gain
    impedance   the same data, all roots of the loop impedance
    phasor      the full plant model with network and stator transients dropped

Each screen gives a score that is larger for cases it considers more at risk. The first
two are defined here. The next two are columns of the T1 dataset (step 08). The last is
the quasi-static reduction of the state matrix (step 06).
"""

from __future__ import annotations

import numpy as np

from .data import t1
from .physics.induction import IGData
from .solver.linearize import quasi_static

# States made algebraic in a positive-sequence (phasor-domain) model of the 25-state
# plant: stator and filter fluxes, capacitor voltage and the loop flux of the network.
PHASOR_FAST = [0, 1, 4, 5, 6, 7, 23, 24]


def network_resonance(
    theta: np.ndarray, x_machine: float | None = None, lossless: bool = False
) -> np.ndarray:
    """Frequency (pu of synchronous) at which the reactance of the loop formed by the
    network and the machine leakage passes from capacitive to inductive. Zero if it never
    does below synchronous frequency. Machine resistance and controls play no part.

    theta has shape (n, 12) in the order of `t1.NAMES`. The network impedances are scaled
    in frequency as in a frequency-scan study: R + j w X_L and -j X_C / w. With
    `lossless` the network resistances are dropped as well.
    """
    base = IGData()
    col = {name: theta[:, j] for j, name in enumerate(t1.NAMES)}
    x_e = col["n_online"] * col["x_line"] * base.x_e
    x_c = col["k"] * x_e
    x_p = col["par_ratio"] * x_e
    r_e = np.zeros_like(x_e) if lossless else x_e / col["xr_line"]
    r_p = col["par_ratio"] * r_e
    x_series = col["x_t"] + (base.x_ls + base.x_lr if x_machine is None else x_machine)
    w = np.linspace(0.01, 1.0, 496)[None, :]
    line = r_e[:, None] + 1j * (w * x_e[:, None] - x_c[:, None] / w)
    path = r_p[:, None] + 1j * w * x_p[:, None]
    x_loop = w * x_series[:, None] + (line * path / (line + path)).imag
    up = (x_loop[:, :-1] < 0) & (x_loop[:, 1:] >= 0)
    first = up.argmax(axis=1)
    rows = np.arange(len(theta))
    x0, x1 = x_loop[rows, first], x_loop[rows, first + 1]
    w0, w1 = w[0, first], w[0, first + 1]
    w_res = w0 - x0 * (w1 - w0) / (x1 - x0)
    return np.where(up.any(axis=1), w_res, 0.0)


def slip_score(theta: np.ndarray) -> np.ndarray:
    """1 / |slip at the network resonance|: how strongly the rotor resistance is
    reflected as negative resistance. Zero without a resonance below rotor speed."""
    w_res = network_resonance(theta)
    w_r = theta[:, t1.NAMES.index("w_r")]
    ok = (w_res > 0) & (w_res < w_r)
    return np.where(ok, w_res / np.maximum(w_r - w_res, 1e-9), 0.0)


def phasor_growth(a: np.ndarray) -> float:
    """Largest growth rate (1/s) of the phasor-domain model: the state matrix with
    network and stator transients made algebraic."""
    return float(np.linalg.eigvals(quasi_static(a, PHASOR_FAST)[0]).real.max())


def zero_miss_threshold(score: np.ndarray, unstable: np.ndarray) -> float:
    """The largest threshold that leaves no unstable case below it: the lowest score of
    any unstable case."""
    return float(score[unstable].min()) if unstable.any() else float("inf")


def screen(score: np.ndarray, sigma: np.ndarray, threshold: float) -> dict[str, float]:
    """Outcome of clearing every case whose score is below the threshold.

    cleared          share of all cases cleared, which need no further study
    missed_unstable  share of unstable cases cleared
    false_alarm      share of stable cases not cleared
    """
    unstable = sigma > 0
    cleared = score < threshold
    return {
        "cleared": float(cleared.mean()),
        "missed_unstable": float(cleared[unstable].mean()) if unstable.any() else float("nan"),
        "false_alarm": float((~cleared)[~unstable].mean()) if (~unstable).any() else float("nan"),
    }
