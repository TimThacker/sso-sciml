"""The converter screens of steps 17 to 24 once more, with a control delay in the
converter.

Each case of dataset C1 is evaluated twice, as it is and with the delay of step 28 (the
same draw: delay times current-loop bandwidth between 0.02 and 0.10). No filter capacitor,
so the comparison is of the band the earlier steps worked in.

Used by scripts/restate_with_delay.py.
"""

from __future__ import annotations

import numpy as np

from . import admittance as adm
from . import hf
from . import vectorfit as vf
from .data import c1
from .data.t1 import case_seed
from .neterror import FACTORS, perturb
from .scanfit import band_growth, noisy

VARIANTS = ("base", "delay")
NOISE, BOUND = 0.01, 0.10  # the reference screen of step 24
COLUMNS = (
    "t_d",
    *(f"{name}_{v}" for v in VARIANTS
      for name in ("solved", "top", "poles", *adm.SCREEN_COLUMNS, "fit", "ref_given", "ref_ends")),
)  # fmt: skip
_GRID = adm.scan_grid()
_S = 2j * np.pi * _GRID
_HALF = _GRID >= 0
_SIGNS = np.array(np.meshgrid(*[(-1.0, 1.0)] * len(FACTORS))).reshape(len(FACTORS), -1).T


def delay_of(theta: np.ndarray) -> float:
    """The delay step 28 drew for this case, s."""
    rng = np.random.default_rng(case_seed(theta) + 28)
    rho = float(np.exp(rng.uniform(*np.log(hf.RANGES["rho"]))))
    return rho / float(theta[c1.NAMES.index("cc_hz")])


def case(args: tuple[str, np.ndarray]) -> np.ndarray:
    """One case in the order of COLUMNS."""
    kind, theta = args
    follow = float(kind == "follow")
    t_d = delay_of(theta)
    n = 3 + len(adm.SCREEN_COLUMNS) + 3
    out = [t_d]
    for variant in VARIANTS:
        try:
            plant, net = hf.solve(theta, follow, t_d if variant == "delay" else 0.0, 0.0, 0.0)
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            out += [0.0] + [np.nan] * (n - 1)
            continue
        zero = np.zeros((2, 2))
        top = float(vf.closed_loop_modes(plant.a, plant.b, plant.c, zero, net).real.max())
        y, z = plant.admittance(_S), adm.network_impedance(net, _S)
        seed = case_seed(theta)
        scores = adm.screen_scores(y, z, plant.unstable_poles, seed)
        step = adm.NOISE_STEP
        fit = adm.fit_growth(y[::step], z[::step], _GRID[::step])[0]
        # Reference screen: a noisy scan fitted, closed on the network as given and at
        # the 32 ends of its ranges.
        rng = np.random.default_rng(seed + 29)
        model = vf.fit_admittance(_GRID[_HALF][::step], noisy(y[_HALF][::step], NOISE, rng), NOISE)

        def modes(other: dict, model=model) -> float:
            lam = vf.closed_loop_modes(model["a"], model["b"], model["c"], model["d"], other)
            return max(band_growth(lam))

        width = np.log1p(BOUND)
        given = perturb(net, np.exp(rng.uniform(-width, width, size=len(FACTORS))))
        ends = max(modes(perturb(given, np.exp(width * s))) for s in _SIGNS)
        out += [1.0, top, plant.unstable_poles, *scores, fit, modes(given), ends]
    return np.array(out, dtype=float)


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def cases(kind: str, theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """`case` for every row of theta. Row order is kept."""
    jobs = [(kind, row) for row in theta]
    if processes <= 1:
        return np.array([case(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(case, jobs, chunksize=32))
