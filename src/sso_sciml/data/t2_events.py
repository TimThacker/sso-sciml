"""Task T2 with three more disturbances: a compensation step, a voltage dip, a power ramp.

The cases, plant, protection, window and observed quantities are those of `t2`. Only
the disturbance differs. The plant starts at its steady state, and at t = 0

    compensation_step   the series capacitor's reactance rises by STEP_K of the line
                        reactance (one more segment in service) and stays there
    voltage_dip         the infinite bus drops to 1 - DIP_DEPTH for DIP_S seconds
    power_ramp          the mechanical power rises by RAMP_PU of rating over RAMP_S
                        seconds, then stays (a gust)

The reduced model of `t2` (machine, rotor current loop and line) is driven by the same
change where it has the input: the capacitor step and the voltage dip. It has no
mechanical power and no outer loops, so its response to the ramp is zero. That is what
the physics given to arm F knows about a gust.
"""

from __future__ import annotations

import numpy as np
import torch

from ..physics.dfig import DFIG, gains_from_bandwidth
from ..physics.dfig_full import DFIGFull
from ..physics.induction import IGData
from . import t2

EVENTS = ("compensation_step", "voltage_dip", "power_ramp")
STEP_K = 0.05
DIP_DEPTH, DIP_S = 0.15, 0.05
RAMP_PU, RAMP_S = 0.10, 0.2
VERSION = "t2e-v1"


def inputs(event: str, t: float, q: dict) -> dict:
    """Parameter entries that change with time, for a batch, at time t (s). `q` holds
    the steady values; `x_e` is the line reactance."""
    if event == "compensation_step":
        return {"x_c": q["x_c"] + STEP_K * q["x_e"]} if t > 0 else {}
    if event == "voltage_dip":
        return {"v_b": q["v_b"] * (1.0 - DIP_DEPTH)} if 0 < t <= DIP_S else {}
    if event == "power_ramp":
        return {"p_m": q["p_m"] + RAMP_PU * min(t / RAMP_S, 1.0)} if t > 0 else {}
    raise ValueError(event)


def _rk4(model, x: torch.Tensor, q: dict, event: str, observe=None, trip=None):
    """RK4 over the window with the event's inputs at each stage time. Returns the saved
    deviations of the states, of `observe` if given, and the trip count."""
    dt = t2.DT

    def f(xx, t):
        return model.rhs(xx, {**q, **inputs(event, t, q)})

    eq = x.clone()
    obs_eq = observe(model, eq, q) if observe else None
    states = [(x - eq).float().cpu()]
    obs = [(observe(model, x, q) - obs_eq).float().cpu()] if observe else []
    valid = torch.full((x.shape[0],), t2.N_T, dtype=torch.long, device=x.device)
    for k in range(1, int(round(t2.T_END / dt)) + 1):
        t = (k - 1) * dt
        k1 = f(x, t)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if trip is not None:
            over = ~(trip(model, x, q) <= t2.TRIPS["rotor current"][2])
            valid = torch.where(over, torch.clamp(valid, max=(k - 1) // t2.EVERY + 1), valid)
        if k % t2.EVERY == 0:
            states.append((x - eq).float().cpu())
            if observe:
                obs.append((observe(model, x, {**q, **inputs(event, k * dt, q)}) - obs_eq)
                           .float().cpu())  # fmt: skip
    out = torch.stack(states, 1).numpy()
    return out, (torch.stack(obs, 1).numpy() if observe else None), valid.cpu().numpy()


@torch.no_grad()
def simulate(x_eq: np.ndarray, p: dict, event: str, device: str = "cuda"):
    """The full plant from its steady state through the event, for a batch. As
    `t2.simulate`: state deviations, observed deviations and trip counts."""
    dev = torch.device(device)
    q = {k: torch.as_tensor(v, dtype=torch.float64, device=dev) for k, v in p.items()}
    x = torch.as_tensor(x_eq, dtype=torch.float64, device=dev)
    return _rk4(DFIGFull(), x, q, event, observe=t2.observe, trip=t2._rotor_current)


def reduced_setup(theta: np.ndarray) -> tuple[np.ndarray, dict]:
    """Steady state and parameters of the reduced model of one case."""
    th = dict(zip(t2.NAMES, map(float, theta), strict=True))
    data = IGData()
    kp, ki = gains_from_bandwidth(data, th["rsc_hz"])
    model = DFIG(data=data)
    x0, p = model.equilibrium(th["k"] * data.x_e, w_r=th["w_r"], kp=kp, ki=ki)
    return x0.numpy(), {k: float(v) for k, v in p.items()}


@torch.no_grad()
def reduced_response(theta: np.ndarray, event: str, device: str = "cuda") -> np.ndarray:
    """Stator current and capacitor voltage of the reduced model through the event, as
    deviations, (n, N_T, 4). Zero for the power ramp, which it has no input for."""
    n = theta.shape[0]
    if event == "power_ramp":
        return np.zeros((n, t2.N_T, 4), dtype=np.float32)
    rows = [reduced_setup(th) for th in theta]
    dev = torch.device(device)
    x = torch.as_tensor(np.array([r[0] for r in rows]), dtype=torch.float64, device=dev)
    q = {k: torch.as_tensor(np.array([r[1][k] for r in rows]), dtype=torch.float64, device=dev)
         for k in rows[0][1]}  # fmt: skip
    q["x_e"] = torch.full((n,), IGData().x_e, dtype=torch.float64, device=dev)
    model = DFIG(data=IGData())
    params = {k: v for k, v in q.items() if k != "x_e"} | {"x_e": q["x_e"]}
    states, _, _ = _rk4(model, x, params, event)
    return (states.astype(np.float64) @ t2.reduced_outputs().T).astype(np.float32)


def reduced_matrix_after(theta: np.ndarray, event: str) -> np.ndarray:
    """State matrix of the reduced model after the event: the new compensation for the
    step, the steady one otherwise."""
    if event != "compensation_step":
        return t2.reduced_matrix(theta)
    th = theta.copy()
    th[0] = th[0] + STEP_K
    return t2.reduced_matrix(th)


def generate(theta: np.ndarray, event: str, processes: int = 8, device: str = "cuda",
             batch: int = 4096) -> dict:  # fmt: skip
    """Every row of theta through the event, with the same keys as `t2.generate`."""
    rows = t2.setups(theta, processes)
    solved = np.array([r["solved"] for r in rows]) > 0
    n = len(rows)
    states = np.full((n, t2.N_T, t2.N_STATE), np.nan, dtype=np.float32)
    obs = np.full((n, t2.N_T, len(t2.OBSERVED)), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=np.int64)
    reduced = np.full((n, t2.N_T, 4), np.nan, dtype=np.float32)
    a_red = np.full((n, 8, 8), np.nan)
    idx = np.where(solved)[0]
    for part in np.array_split(idx, max(1, int(np.ceil(idx.size / batch)))):
        p = {k: np.array([rows[i]["p"][k] for i in part]) for k in rows[part[0]]["p"]}
        x_eq = np.array([rows[i]["x_eq"] for i in part])
        states[part], obs[part], valid[part] = simulate(x_eq, p, event, device)
        reduced[part] = reduced_response(theta[part], event, device)
    for i in idx:
        a_red[i] = reduced_matrix_after(theta[i], event)
    after = np.arange(t2.N_T)[None, :] >= valid[:, None]
    states[after], obs[after] = 0.0, 0.0
    solved &= np.isfinite(states).all(axis=(1, 2)) & (valid >= t2.N_T // 4)
    return {"theta": theta, "solved": solved, "states": states, "observed": obs, "valid": valid,
            "x_eq": np.array([r["x_eq"] for r in rows]), "reduced": reduced, "a_red": a_red,
            "sigma": np.array([r["sigma"] for r in rows]),
            "f": np.array([r["f"] for r in rows])}  # fmt: skip
