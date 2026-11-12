"""Task T1 dataset: parameters to mechanism quantities and stability of System B2.

One case is a point in a 12-dimensional parameter space. For each case this module
returns

    - solver outputs: growth rate and frequency of the sub-synchronous mode and of the
      least-damped mode, from the full-control model on the meshed network (25 states),
    - mechanism quantities from the hand calculation: resonance frequency, slip at
      resonance, machine and net resistance, the quick growth-rate estimate, and the
      root of the equivalent circuit.

The hand calculation knows the machine, the rotor current loop and the network. It does
not know the phase-locked loop, the outer loops or the grid-side converter. The gap
between the two is what a learned model has to supply.

Ranges, sampling and splits are fixed by `SPEC` and `SPLITS`. Changing either changes
`VERSION`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.stats import qmc

from .. import admittance as adm
from ..physics import circuit, slip
from ..physics.dfig import gains_from_bandwidth
from ..physics.dfig_full import Controls
from ..physics.dfig_meshed import DFIGMeshed
from ..physics.induction import IGData
from ..solver.linearize import jacobian, modes

VERSION = "t1-v2"
SEED = 20261111
XR_PLANT = 15.0  # X/R of the plant-side impedance


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    log: bool
    meaning: str
    source: str


SPEC: tuple[Param, ...] = (
    Param("k", 0.10, 0.80, False, "series compensation as a fraction of line reactance",
          "practical installations run from about 20 to 75 %"),
    Param("w_r", 0.80, 1.20, False, "rotor speed, pu",
          "operating range of a doubly-fed machine, about +/-30 % slip, less a margin"),
    Param("rsc_hz", 2.0, 30.0, True, "rotor current loop bandwidth, Hz",
          "step 05: stability limit lies between 3 and 32 Hz across networks"),
    Param("pll_hz", 5.0, 30.0, True, "phase-locked loop natural frequency, Hz",
          "representative; step 04 swept 1 to 60 Hz"),
    Param("outer_ratio", 0.03, 0.15, True, "outer power loop crossover over current loop bandwidth",
          "cascade tuning; sampled independently the two loops interact below 2 Hz"),
    Param("ff", 0.0, 1.0, False, "terminal-voltage feed-forward gain of the grid-side converter",
          "step 04: off and on bracket practice"),
    Param("x_line", 0.5, 1.5, False, "line reactance as a multiple of the reference 0.392 pu",
          "line length; representative"),
    Param("xr_line", 10.0, 30.0, True, "line X/R",
          "reference line is 23.3; step 05 varied resistance 0.5 to 8 times"),
    Param("x_t", 0.05, 0.25, False, "plant-side series reactance, pu on the plant base",
          "transformers and collector; representative (A27)"),
    Param("par_ratio", 0.5, 8.0, True, "parallel path impedance as a multiple of the line's",
          "step 05 varied 0.25 to 32"),
    Param("n_online", 0.3, 1.0, False, "fraction of units online",
          "step 05: worst case between 35 and 65 %"),
    Param("r_r", 0.004, 0.008, False, "rotor resistance, pu",
          "around the reference 0.00549 (A12)"),
)  # fmt: skip
NAMES = tuple(p.name for p in SPEC)

# Extrapolation tests: one parameter leaves its training range, the others stay inside.
SPLITS: dict[str, dict[str, tuple[float, float]]] = {
    "in_distribution": {},
    "high_compensation": {"k": (0.80, 0.95)},
    "low_speed": {"w_r": (0.70, 0.80)},
    "fast_current_loop": {"rsc_hz": (30.0, 60.0)},
    "weak_parallel_path": {"par_ratio": (8.0, 64.0)},
    "other_machine": {"r_r": (0.008, 0.012)},
}

SOLVER_COLUMNS = ("sigma_sso", "f_sso", "sigma_max", "f_max", "solved")
MECHANISM_COLUMNS = (
    "has_resonance", "f_er", "slip_res", "r_machine", "r_net", "sigma_hand", "sigma_circuit",
    "f_circuit",
    "sigma_impedance",
    "f_impedance",
)  # fmt: skip


def spec_hash() -> str:
    blob = json.dumps(
        {"version": VERSION, "seed": SEED, "spec": [vars(p) for p in SPEC], "splits": SPLITS},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sample(n: int, split: str = "in_distribution", stream: int = 0) -> np.ndarray:
    """Scrambled Sobol points scaled to the ranges of a split. Shape (n, 12).

    The same (n, split, stream) always returns the same array.
    """
    seed = SEED + 1000 * list(SPLITS).index(split) + stream
    unit = qmc.Sobol(d=len(SPEC), scramble=True, seed=seed).random(n)
    out = np.empty_like(unit)
    for j, par in enumerate(SPEC):
        lo, hi = SPLITS[split].get(par.name, (par.low, par.high))
        if par.log:
            out[:, j] = np.exp(np.log(lo) + unit[:, j] * (np.log(hi) - np.log(lo)))
        else:
            out[:, j] = lo + unit[:, j] * (hi - lo)
    return out


def _network(th: dict, base: IGData) -> dict:
    """Network impedances on the base of the machines that are running."""
    x_e = th["n_online"] * th["x_line"] * base.x_e
    r_e = x_e / th["xr_line"]
    return {
        "x_e": x_e, "r_e": r_e, "x_c": th["k"] * x_e,
        "x_p": th["par_ratio"] * x_e, "r_p": th["par_ratio"] * r_e,
        "x_t": th["x_t"], "r_t": th["x_t"] / XR_PLANT,
    }  # fmt: skip


def mechanism(theta: np.ndarray) -> np.ndarray:
    """Hand-calculation quantities for one case, in the order of MECHANISM_COLUMNS."""
    th = dict(zip(NAMES, map(float, theta), strict=True))
    data = replace(IGData(), r_r=th["r_r"])
    net = _network(th, data)
    kp, ki = gains_from_bandwidth(data, th["rsc_hz"])

    def z_network(s: complex) -> complex:
        z_c = net["r_e"] + s * net["x_e"] + net["x_c"] / s
        z_p = net["r_p"] + s * net["x_p"]
        return net["r_t"] + s * net["x_t"] + z_c * z_p / (z_c + z_p)

    def z_machine(s: complex) -> complex:
        return slip.dfig_impedance(
            s, data.r_s, data.x_ls, data.x_m, data.r_r, data.x_lr, th["w_r"], kp, ki
        )

    # Impedance model of step 05: all roots of the loop impedance as polynomials. Unlike
    # the resonance search below it has an answer in every case. Of the modes between 2
    # and 58 Hz in the rotating frame, the least damped is taken.
    z_poly = circuit.parallel(
        circuit.branch(net["r_e"], net["x_e"], net["x_c"]), circuit.branch(net["r_p"], net["x_p"])
    ) + circuit.branch(net["r_t"], net["x_t"])
    lam = circuit.modes(
        circuit.dfig_impedance(
            data.r_s, data.x_ls, data.x_m, data.r_r, data.x_lr, th["w_r"], kp, ki
        )
        + z_poly
    )
    f_lam = np.abs(lam.imag) / (2 * np.pi)
    band = np.where((f_lam > 2.0) & (f_lam < 58.0))[0]
    if band.size:
        top = band[np.argmax(lam.real[band])]
        imp = [float(lam.real[top]), float(f_lam[top])]
    else:
        imp = [0.0, 0.0]

    try:
        # Stay clear of synchronous frequency, where integral action adds crossings.
        r = slip.analyse_loop(
            z_machine, 0.0, 0.0, 0.0, w_r=th["w_r"], z_network=z_network,
            w_max=min(th["w_r"], 1.0) - 0.05,
        )  # fmt: skip
    except ValueError:
        return np.array([0.0] * 8 + imp)
    return np.array(
        [1.0, r.f_er, r.slip, r.r_machine, r.r_net, r.sigma_hand, r.sigma_circuit, r.f_circuit]
        + imp
    )


N_STATE = 25


def operating_point(theta: np.ndarray) -> tuple[DFIGMeshed, torch.Tensor, dict]:
    """The full model of one case with its steady state and parameter dictionary.
    Raises ValueError if the steady state is not found."""
    th = dict(zip(NAMES, map(float, theta), strict=True))
    data = replace(IGData(), r_r=th["r_r"])
    net = _network(th, data)
    ctl = Controls(rsc_hz=th["rsc_hz"], pll_hz=th["pll_hz"], ff=th["ff"],
                   outer_hz=th["outer_ratio"] * th["rsc_hz"])  # fmt: skip
    model = DFIGMeshed(data=data, controls=ctl)
    x0, p = model.equilibrium(net.pop("x_c"), w_r=th["w_r"], **net)
    if float(model.rhs(x0, p).abs().max()) > 1e-8:
        raise ValueError("steady state not found")
    return model, x0, p


def state_matrix(theta: np.ndarray) -> np.ndarray:
    """State matrix A(theta) of the full model at its steady state, shape (25, 25), in
    1/s. NaN if the steady state is not found. This is the solver without its last step,
    the eigenvalue decomposition."""
    try:
        model, x0, p = operating_point(theta)
        return jacobian(model.rhs, x0, p).numpy()
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return np.full((N_STATE, N_STATE), np.nan)


def state_matrices(theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """A(theta) for every row of theta, shape (n, 25, 25). Row order is kept."""
    if processes <= 1:
        return np.array([state_matrix(row) for row in theta])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(state_matrix, list(theta), chunksize=64))


def solve(theta: np.ndarray) -> np.ndarray:
    """Full-solver outputs for one case, in the order of SOLVER_COLUMNS."""
    a = state_matrix(theta)
    try:
        if not np.isfinite(a).all():
            raise ValueError("steady state not found")
        md = modes(a)
    except (ValueError, np.linalg.LinAlgError):
        return np.array([np.nan, np.nan, np.nan, np.nan, 0.0])
    cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 58.0))[0]
    j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
    top = int(np.argmax(md.sigma))
    return np.array([md.sigma[j], abs(md.freq_hz[j]), md.sigma[top], abs(md.freq_hz[top]), 1.0])


SCAN_COLUMNS = ("solved", "sigma", "f", "poles", "f_passive", "g_min", *adm.SCREEN_COLUMNS)
_GRID = adm.scan_grid()
_S = 2j * np.pi * _GRID


def case_seed(theta: np.ndarray) -> int:
    """A seed fixed by the parameters of a case, for the noise added to its scan."""
    return int.from_bytes(hashlib.sha256(np.ascontiguousarray(theta).tobytes()).digest()[:8], "big")


def network(theta: np.ndarray) -> dict:
    """Network impedances of one case, with the keys `admittance.network_impedance` uses."""
    th = dict(zip(NAMES, map(float, theta), strict=True))
    return _network(th, replace(IGData(), r_r=th["r_r"]))


def scan_pair(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Plant admittance and network impedance of one case on `admittance.scan_grid()`,
    and the number of unstable poles of the plant alone."""
    model, x0, p = operating_point(theta)
    plant = adm.DFIGScan(model, x0, p)
    return plant.admittance(_S), adm.network_impedance(network(theta), _S), plant.unstable_poles


def scan(theta: np.ndarray) -> np.ndarray:
    """One case as a 2 x 2 admittance scan of the plant against the impedance of its
    network, in the order of SCAN_COLUMNS. `sigma` and `f` are the largest growth rate
    of any mode of the full model (1/s) and its frequency (Hz, synchronous frame)."""
    out = np.full(len(SCAN_COLUMNS), np.nan)
    out[0] = 0.0
    try:
        model, x0, p = operating_point(theta)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return out
    lam = np.linalg.eigvals(jacobian(model.rhs, x0, p).numpy())
    top = int(np.argmax(lam.real))
    plant = adm.DFIGScan(model, x0, p)
    y, z = plant.admittance(_S), adm.network_impedance(network(theta), _S)
    index = adm.passivity_index(y)
    bad = index < 0
    out[:] = [
        1.0, lam.real[top], abs(lam.imag[top]) / (2 * np.pi), plant.unstable_poles,
        float(np.abs(_GRID[bad]).max()) if bad.any() else 0.0, float(index.min()),
        *adm.screen_scores(y, z, plant.unstable_poles, case_seed(theta)),
    ]  # fmt: skip
    return out


def scans(theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """`scan` for every row of theta. Row order is kept."""
    if processes <= 1:
        return np.array([scan(row) for row in theta])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(scan, list(theta), chunksize=64))


def evaluate(theta: np.ndarray) -> np.ndarray:
    """Solver outputs followed by mechanism quantities for one case."""
    return np.concatenate([solve(theta), mechanism(theta)])


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def generate(theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """Evaluate every row of theta. Row order is kept, so the result does not depend on
    the number of processes."""
    if processes <= 1:
        return np.array([evaluate(row) for row in theta])
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return np.array(pool.map(evaluate, list(theta), chunksize=64))


def _bisect(theta: np.ndarray, steps: int = 6) -> np.ndarray:
    """Move one case along compensation to the stability boundary of the sub-synchronous
    mode, if there is one inside the training range. Returns the case, or NaN."""
    k_col = NAMES.index("k")
    lo, hi = theta.copy(), theta.copy()
    lo[k_col], hi[k_col] = SPEC[k_col].low, SPEC[k_col].high
    s_lo, s_hi = solve(lo)[0], solve(hi)[0]
    if not (np.isfinite(s_lo) and np.isfinite(s_hi)) or s_lo * s_hi > 0:
        return np.full_like(theta, np.nan)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        s_mid = solve(mid)[0]
        if not np.isfinite(s_mid):
            return np.full_like(theta, np.nan)
        if s_mid * s_lo > 0:
            lo, s_lo = mid, s_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def boundary_cases(theta: np.ndarray, processes: int = 1) -> np.ndarray:
    """Cases moved to within 0.6 points of compensation of the boundary. Rows without a
    boundary in range are dropped."""
    if processes <= 1:
        out = np.array([_bisect(row) for row in theta])
    else:
        from multiprocessing import Pool

        with Pool(processes, initializer=_init_worker) as pool:
            out = np.array(pool.map(_bisect, list(theta), chunksize=16))
    return out[np.isfinite(out[:, 0])]


def folds(n: int) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, in fixed proportions 80 / 10 / 10."""
    order = np.random.default_rng(SEED).permutation(n)
    fold = np.zeros(n, dtype=np.int8)
    fold[order[int(0.8 * n) : int(0.9 * n)]] = 1
    fold[order[int(0.9 * n) :]] = 2
    return fold
