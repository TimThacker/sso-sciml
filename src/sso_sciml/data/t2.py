"""Task T2 dataset: parameters to the response in time of System B2.

One case is a point in a 4-dimensional parameter space: series compensation, rotor speed,
bandwidth of the rotor current loop and natural frequency of the phase-locked loop. The
plant is the doubly-fed machine with its full controls on the reference series line
(`physics.dfig_full`, 23 states), with the limits and the chopper of step 33 in. The
series capacitor voltage is put OFFSET off its steady value and the plant is run for
T_END seconds.

For each case this module returns

    - the deviation of every state from its steady value at N_T times,
    - five observed quantities: line current (D, Q), capacitor voltage (D, Q), DC voltage,
    - the same four network quantities from the reduced model of `physics.dfig`, which
      knows the machine, the rotor current loop and the line and nothing else. It is
      linear, so its response is written down from its modes with no time stepping,
    - the growth rate and frequency of the sub-synchronous mode of the full model,
    - the number of saved times before the rotor current passes its trip level of
      step 33. Later values are zeroed and not used. A case that trips in the first
      quarter of the run is dropped.

The reduced model plays the part for T2 that the hand calculation plays for T1.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from ..physics.dfig import DFIG, gains_from_bandwidth
from ..physics.dfig_full import Controls, DFIGFull
from ..physics.induction import IGData
from ..protection import SETTINGS, TRIPS
from ..solver.linearize import jacobian, modes
from ..solver.modal import linear_response
from .t1 import Param

VERSION = "t2-v1"
SEED = 20261113
OFFSET = 0.01  # pu, on the D component of the capacitor voltage
T_END, DT, EVERY = 0.3, 5e-4, 3
N_T = int(round(T_END / DT)) // EVERY + 1
OUTER_RATIO = 0.07  # outer loop crossover over current loop bandwidth
N_STATE = 23
VC = 4
OBSERVED = ("il_D", "il_Q", "vc_D", "vc_Q", "v_dc")

SPEC: tuple[Param, ...] = (
    Param("k", 0.05, 0.50, False, "series compensation as a fraction of line reactance",
          "the boundary of the reference line lies between 12 and 31 % (step 05)"),
    Param("w_r", 0.80, 1.20, False, "rotor speed, pu", "as in dataset T1"),
    Param("rsc_hz", 2.0, 30.0, True, "rotor current loop bandwidth, Hz", "as in dataset T1"),
    Param("pll_hz", 5.0, 30.0, True, "phase-locked loop natural frequency, Hz",
          "as in dataset T1"),
)  # fmt: skip
NAMES = tuple(p.name for p in SPEC)
SPLITS: dict[str, dict[str, tuple[float, float]]] = {
    "in_distribution": {},
    "high_compensation": {"k": (0.50, 0.60)},
    "low_speed": {"w_r": (0.70, 0.80)},
    "fast_current_loop": {"rsc_hz": (30.0, 60.0)},
}


def spec_hash() -> str:
    blob = json.dumps(
        {"version": VERSION, "seed": SEED, "spec": [vars(p) for p in SPEC], "splits": SPLITS,
         "offset": OFFSET, "t_end": T_END, "dt": DT, "every": EVERY, "protection": SETTINGS,
         "outer_ratio": OUTER_RATIO},
        sort_keys=True,
    )  # fmt: skip
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sample(n: int, split: str = "in_distribution") -> np.ndarray:
    """Scrambled Sobol points scaled to the ranges of a split. Shape (n, 4)."""
    from scipy.stats import qmc

    seed = SEED + 1000 * list(SPLITS).index(split)
    unit = qmc.Sobol(d=len(SPEC), scramble=True, seed=seed).random(n)
    out = np.empty_like(unit)
    for j, par in enumerate(SPEC):
        lo, hi = SPLITS[split].get(par.name, (par.low, par.high))
        if par.log:
            out[:, j] = np.exp(np.log(lo) + unit[:, j] * (np.log(hi) - np.log(lo)))
        else:
            out[:, j] = lo + unit[:, j] * (hi - lo)
    return out


def scale_theta(theta: np.ndarray) -> np.ndarray:
    """Parameters to about [-1, 1] over the training range, in log where sampled in log."""
    out = np.empty_like(theta, dtype=float)
    for j, par in enumerate(SPEC):
        v, lo, hi = theta[:, j], par.low, par.high
        if par.log:
            v, lo, hi = np.log(v), np.log(lo), np.log(hi)
        out[:, j] = (2 * v - (lo + hi)) / (hi - lo)
    return out


def times() -> np.ndarray:
    return np.arange(N_T) * (DT * EVERY)


def plant(theta: np.ndarray) -> tuple[DFIGFull, float, float]:
    """The full model of one case, its series capacitor reactance and rotor speed."""
    th = dict(zip(NAMES, map(float, theta), strict=True))
    ctl = Controls(rsc_hz=th["rsc_hz"], pll_hz=th["pll_hz"],
                   outer_hz=OUTER_RATIO * th["rsc_hz"], **SETTINGS)  # fmt: skip
    model = DFIGFull(controls=ctl)
    return model, th["k"] * model.data.x_e, th["w_r"]


def reduced_matrix(theta: np.ndarray) -> np.ndarray:
    """State matrix (8, 8) of the reduced model of one case, 1/s. The model is linear, so
    this does not depend on the operating point."""
    th = dict(zip(NAMES, map(float, theta), strict=True))
    data = IGData()
    kp, ki = gains_from_bandwidth(data, th["rsc_hz"])
    model = DFIG(data=data)
    x0, p = model.equilibrium(th["k"] * data.x_e, w_r=th["w_r"], kp=kp, ki=ki)
    return jacobian(model.rhs, x0, p).numpy()


def reduced_outputs() -> np.ndarray:
    """Rows that take the reduced state to stator current (D, Q) and capacitor voltage
    (D, Q). Stator current is a fixed combination of the two fluxes."""
    d = IGData()
    x_ss, x_rr = d.x_ls + d.x_m + d.x_e, d.x_lr + d.x_m
    det = d.x_m**2 - x_ss * x_rr
    out = np.zeros((4, 8))
    out[0, 0], out[0, 2] = x_rr / det, -d.x_m / det
    out[1, 1], out[1, 3] = x_rr / det, -d.x_m / det
    out[2, 4] = out[3, 5] = 1.0
    return out


def reduced_response(theta: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """Stator current and capacitor voltage of the reduced model after the offset, as
    deviations, shape (len(t), 4)."""
    dx0 = np.zeros(8)
    dx0[VC] = OFFSET
    dx = linear_response(reduced_matrix(theta), dx0, times() if t is None else t)
    return dx @ reduced_outputs().T


def setup(theta: np.ndarray) -> dict:
    """Everything about one case that needs no time stepping. `solved` is 0 without a
    steady state."""
    nan = np.nan
    out = {"solved": 0.0, "x_eq": np.full(N_STATE, nan), "sigma": nan, "f": nan, "p": None,
           "reduced": np.full((N_T, 4), nan), "a_red": np.full((8, 8), nan)}  # fmt: skip
    try:
        model, x_c, w_r = plant(theta)
        x0, p = model.equilibrium(x_c, w_r=w_r)
        if float(model.rhs(x0, p).abs().max()) > 1e-8:
            return out
        md = modes(jacobian(model.rhs, x0, p).numpy())
        cand = np.where((md.freq_hz > 2.0) & (md.freq_hz < 58.0))[0]
        j = int(cand[np.argmax(md.participation[4:6].sum(axis=0)[cand])])
        a_red = reduced_matrix(theta)
        dx0 = np.zeros(8)
        dx0[VC] = OFFSET
        out.update(solved=1.0, x_eq=x0.numpy(), sigma=float(md.sigma[j]),
                   f=float(abs(md.freq_hz[j])), p={k: float(v) for k, v in p.items()},
                   reduced=linear_response(a_red, dx0, times()) @ reduced_outputs().T,
                   a_red=a_red)  # fmt: skip
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    return out


def reduced_mode(a_red: np.ndarray) -> np.ndarray:
    """Growth rate (1/s) and frequency (Hz) of the least-damped mode of each reduced
    state matrix between 2 and 58 Hz in the synchronous frame. Shape (n, 2)."""
    lam = np.linalg.eigvals(a_red)
    f = np.abs(lam.imag) / (2 * np.pi)
    real = np.where((f > 2.0) & (f < 58.0), lam.real, -np.inf)
    top = real.argmax(axis=1)
    rows = np.arange(len(a_red))
    return np.stack([lam.real[rows, top], f[rows, top]], axis=1)


def observation_matrix() -> np.ndarray:
    """The observed quantities as a fixed combination of the 23 states, shape (5, 23).
    Machine, filter and line reactances are the same in every case, so the currents are
    the same combination of the fluxes in every case."""
    model = DFIGFull()
    x0, p = model.equilibrium(0.2 * model.data.x_e)
    return torch.func.jacrev(lambda x: observe(model, x, p))(x0).numpy()


def _init_worker() -> None:
    torch.set_num_threads(1)


def setups(theta: np.ndarray, processes: int = 1) -> list[dict]:
    if processes <= 1:
        return [setup(row) for row in theta]
    from multiprocessing import Pool

    with Pool(processes, initializer=_init_worker) as pool:
        return pool.map(setup, list(theta), chunksize=64)


def observe(model: DFIGFull, x: torch.Tensor, p: dict) -> torch.Tensor:
    """The observed quantities at states x, shape (..., 5)."""
    c = model.currents(x, p)
    return torch.stack([c["il_D"], c["il_Q"], x[..., 4], x[..., 5], x[..., 19]], dim=-1)


def _rotor_current(model: DFIGFull, x: torch.Tensor, p: dict) -> torch.Tensor:
    c = model.currents(x, p)
    return torch.hypot(c["ir_D"], c["ir_Q"])


@torch.no_grad()
def simulate(x_eq: np.ndarray, p: dict, device: str = "cuda"):
    """RK4 for a batch of cases after the offset. `p` holds one array per parameter.
    Returns the state deviations (batch, N_T, 23), the observed deviations
    (batch, N_T, 5), both float32, and for each case the number of saved times before
    the rotor current passes its trip level (N_T if it never does). A real plant is off
    the network from there on, so later values are not to be used."""
    model = DFIGFull()
    dev = torch.device(device)
    q = {k: torch.as_tensor(v, dtype=torch.float64, device=dev) for k, v in p.items()}
    eq = torch.as_tensor(x_eq, dtype=torch.float64, device=dev)
    x = eq.clone()
    x[:, VC] += OFFSET
    obs_eq = observe(model, eq, q)
    states, obs = [(x - eq).float().cpu()], [(observe(model, x, q) - obs_eq).float().cpu()]
    valid = torch.full((x.shape[0],), N_T, dtype=torch.long, device=dev)
    for k in range(1, int(round(T_END / DT)) + 1):
        k1 = model.rhs(x, q)
        k2 = model.rhs(x + 0.5 * DT * k1, q)
        k3 = model.rhs(x + 0.5 * DT * k2, q)
        k4 = model.rhs(x + DT * k3, q)
        x = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        over = ~(_rotor_current(model, x, q) <= TRIPS["rotor current"][2])
        valid = torch.where(over, torch.clamp(valid, max=(k - 1) // EVERY + 1), valid)
        if k % EVERY == 0:
            states.append((x - eq).float().cpu())
            obs.append((observe(model, x, q) - obs_eq).float().cpu())
    return torch.stack(states, 1).numpy(), torch.stack(obs, 1).numpy(), valid.cpu().numpy()


def generate(theta: np.ndarray, processes: int = 8, device: str = "cuda",
             batch: int = 4096) -> dict:  # fmt: skip
    """Every row of theta. Rows without a steady state, or whose run fails, have
    solved = 0 and NaN."""
    rows = setups(theta, processes)
    solved = np.array([r["solved"] for r in rows]) > 0
    n = len(rows)
    states = np.full((n, N_T, N_STATE), np.nan, dtype=np.float32)
    obs = np.full((n, N_T, len(OBSERVED)), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=np.int64)
    idx = np.where(solved)[0]
    for part in np.array_split(idx, max(1, int(np.ceil(idx.size / batch)))):
        p = {k: np.array([rows[i]["p"][k] for i in part]) for k in rows[part[0]]["p"]}
        x_eq = np.array([rows[i]["x_eq"] for i in part])
        states[part], obs[part], valid[part] = simulate(x_eq, p, device)
    # Values after the trip may be anything; they are set to zero and never used.
    after = np.arange(N_T)[None, :] >= valid[:, None]
    states[after], obs[after] = 0.0, 0.0
    solved &= np.isfinite(states).all(axis=(1, 2)) & (valid >= N_T // 4)
    return {
        "theta": theta, "solved": solved, "states": states, "observed": obs, "valid": valid,
        "x_eq": np.array([r["x_eq"] for r in rows]),
        "reduced": np.array([r["reduced"] for r in rows], dtype=np.float32),
        "a_red": np.array([r["a_red"] for r in rows]),
        "sigma": np.array([r["sigma"] for r in rows]), "f": np.array([r["f"] for r in rows]),
    }  # fmt: skip


def folds(n: int) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, in fixed proportions 80 / 10 / 10."""
    order = np.random.default_rng(SEED).permutation(n)
    fold = np.zeros(n, dtype=np.int8)
    fold[order[int(0.8 * n) : int(0.9 * n)]] = 1
    fold[order[int(0.9 * n) :]] = 2
    return fold
