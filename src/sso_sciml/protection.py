"""Protection of the doubly-fed plant: what happens after the first limit (step 33).

Step 16 limited the rotor voltage and nothing else, and found that the plant ran away
once the limit acted. Here the converter has what a real one has around its limits:

    rotor voltage limit    the command keeps its direction, its size is held (step 16)
    rotor current limit    the same on the current reference
    anti-windup            integrators track the limited command or stop
    DC chopper             a resistor across the link, switched in above a voltage

and the run is watched for the conditions on which a plant is taken off the network. A
trip is read off the run; the plant is not disconnected in the model, so nothing after
the first trip means anything.

All levels are round figures (A39). Used by scripts/protection.py.
"""

from __future__ import annotations

import numpy as np
import torch

# Controller settings with protection in. `chop_g` passes rated power at full conduction.
SETTINGS = {"vr_max": 0.4, "ir_max": 1.5, "aw": 1.0, "chop_on": 1.10, "chop_width": 0.05,
            "chop_g": 0.8}  # fmt: skip
LIMIT_ONLY = {"vr_max": 0.4}  # the model of step 16
SIGNALS = ("v_dc", "i_rotor", "v_terminal", "speed", "v_command", "i_line", "p_chopper",
           "i_reference")  # fmt: skip
# name: (signal, lowest allowed, highest allowed, time outside before the trip, s)
TRIPS = {
    "DC voltage": ("v_dc", 0.75, 1.25, 0.0),
    "rotor current": ("i_rotor", 0.0, 2.0, 0.0),  # where a crowbar would fire
    "terminal voltage": ("v_terminal", 0.8, 1.2, 0.1),
    "speed": ("speed", 0.65, 1.35, 0.0),
}
V_DC, W_R = 19, 8


def with_settings(p: dict, settings: dict) -> dict:
    return {**p, **{k: torch.full_like(p["kp"], v) for k, v in settings.items()}}


def signals(model, x: torch.Tensor, p: dict) -> torch.Tensor:
    """The watched quantities at states x, shape (..., len(SIGNALS))."""
    out = model.outputs(x, p)
    ref_d = p["out_kp"] * (p["k_opt"] * x[..., W_R] ** 3 - x[..., 13]) + p["out_ki"] * x[..., 15]
    ref_q = p["out_kp"] * (p["q_ref"] - x[..., 14]) + p["out_ki"] * x[..., 16]
    return torch.stack([
        x[..., V_DC], torch.hypot(*out["ir"]), torch.hypot(*out["vt"]), x[..., W_R],
        torch.hypot(*out["vr_command"]), torch.hypot(*out["il"]), out["p_chop"],
        torch.hypot(ref_d, ref_q),
    ], dim=-1)  # fmt: skip


@torch.no_grad()
def run(model, x0: torch.Tensor, p: dict, t_end: float, dt: float, every: int = 2):
    """RK4 from x0, shape (batch, n). Returns the times, the watched quantities with shape
    (n_t, batch, len(SIGNALS)) and, per case, the time its integration failed (NaN if it
    did not). A case that fails is held where it was so that the batch can go on."""
    x = x0.clone()
    alive = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    failed = torch.full((x.shape[0],), float("nan"), dtype=torch.float64, device=x.device)
    out = [signals(model, x, p).cpu()]
    n_steps = int(round(t_end / dt))
    for k in range(1, n_steps + 1):
        k1 = model.rhs(x, p)
        k2 = model.rhs(x + 0.5 * dt * k1, p)
        k3 = model.rhs(x + 0.5 * dt * k2, p)
        k4 = model.rhs(x + dt * k3, p)
        new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ok = torch.isfinite(new).all(dim=-1) & (new.abs().amax(dim=-1) < 50.0)
        ok &= new[:, V_DC] > 0.2
        gone = alive & ~ok
        failed[gone] = k * dt
        alive &= ok
        x = torch.where(alive[:, None], new, x)
        if k % every == 0:
            s = signals(model, x, p)
            s[~alive] = float("nan")
            out.append(s.cpu())
    t = np.arange(len(out)) * (dt * every)
    return t, torch.stack(out).numpy(), failed.cpu().numpy()


def first_outside(t: np.ndarray, value: np.ndarray, low: float, high: float, delay: float):
    """First time each column of value (n_t, batch) has been outside [low, high] for
    `delay` seconds without a break; NaN if never. NaN values count as inside."""
    with np.errstate(invalid="ignore"):
        outside = (value < low) | (value > high)
    need = int(round(delay / (t[1] - t[0])))
    count = np.zeros(value.shape[1], dtype=int)
    when = np.full(value.shape[1], np.nan)
    for k in range(len(t)):
        count = np.where(outside[k], count + 1, 0)
        hit = (count > need) & np.isnan(when)
        when[hit] = t[k]
    return when


def trip_times(
    t: np.ndarray, sig: np.ndarray, failed: np.ndarray | None = None, trips: dict | None = None
) -> dict:
    """Time of each trip per case, and of the first of them. A run that failed counts as
    a trip at that time under "run failed". `trips` replaces TRIPS."""
    col = {name: j for j, name in enumerate(SIGNALS)}
    out = {name: first_outside(t, sig[:, :, col[s]], lo, hi, delay)
           for name, (s, lo, hi, delay) in (TRIPS if trips is None else trips).items()}  # fmt: skip
    if failed is not None:
        out["run failed"] = failed
    out["first"] = np.fmin.reduce(list(out.values()))
    return out


def swing(t: np.ndarray, value: np.ndarray, t_from: float, t_to: float) -> np.ndarray:
    """Half the peak-to-peak range of each column of value between two times."""
    m = (t >= t_from) & (t <= t_to)
    return 0.5 * (np.nanmax(value[m], axis=0) - np.nanmin(value[m], axis=0))
