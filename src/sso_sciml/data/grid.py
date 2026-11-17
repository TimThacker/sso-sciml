"""Dataset G1: several doubly-fed plants on a meshed-free (tree) network (step 51).

A case is a tree of `n` buses fed from an infinite bus. Every line is R + L, some with a
series capacitor; at each bus there may be a doubly-fed plant (the current-loop model of
`physics.dfig`, as in the impedance formulation of step 05) behind its transformer. Small
signal, the infinite bus is a short, so the modes are the values of s at which the
nodal admittance matrix of the n buses is singular:

    Y(s)[i, i] = sum of the admittances of the branches at bus i + plant admittance
    Y(s)[i, j] = - admittance of the line between i and j

with s the per-unit stator-frame frequency (circuit.py), lambda = wb (s - j). A
sub-synchronous mode has s near j f_er / f0 for an electrical resonance f_er between 2
and 58 Hz; its rotating-frame frequency is f0 - f_er.

The modes are found by Newton's method on det Y(s) from the minima of |det Y| along
the frequency axis. For one plant on one line this is the root of Z_line + Z_plant of
step 05 (a test).
"""

from __future__ import annotations

import numpy as np

from ..physics import circuit
from ..physics.dfig import gains_from_bandwidth
from ..physics.induction import IGData

VERSION = "g1-v1"
SEED = 20261116
F0 = 60.0
X_T = 0.05  # plant transformer, pu
SCAN_HZ = np.linspace(2.0, 58.0, 225)


def sample(n_bus: int, rng: np.random.Generator) -> dict:
    """One random case with n_bus buses. Bus 0 is the infinite bus; buses 1..n_bus."""
    parent = np.array([rng.integers(0, i) for i in range(1, n_bus + 1)])  # tree
    comp = rng.random(n_bus) < 0.5
    line = {"x": rng.uniform(0.05, 0.4, n_bus), "xr": np.exp(rng.uniform(np.log(10), np.log(30),
                                                                          n_bus)),
            "k": np.where(comp, rng.uniform(0.2, 0.7, n_bus), 0.0)}  # fmt: skip
    has = rng.random(n_bus) < 0.75
    has[rng.integers(0, n_bus)] = True  # at least one plant
    plant = {"has": has, "w_r": rng.uniform(0.8, 1.2, n_bus),
             "rsc_hz": np.exp(rng.uniform(np.log(2.0), np.log(30.0), n_bus)),
             "size": rng.uniform(0.2, 1.0, n_bus)}  # fmt: skip
    return {"n": n_bus, "parent": parent, "line": line, "plant": plant}


def _plant_impedance(w_r: float, rsc_hz: float, size: float) -> circuit.Rational:
    d = IGData()
    kp, ki = gains_from_bandwidth(d, rsc_hz)
    z = circuit.dfig_impedance(d.r_s, d.x_ls, d.x_m, d.r_r, d.x_lr, w_r, kp, ki)
    return (z + circuit.branch(X_T / 15.0, X_T)) * (1.0 / size)


def impedances(case: dict, only: int | None = None) -> tuple[list, list]:
    """Line impedances (one per bus, to its parent) and plant impedances (None where
    there is no plant). With `only`, every plant but that bus's is left out."""
    ln = case["line"]
    lines = [circuit.branch(ln["x"][i] / ln["xr"][i], ln["x"][i], ln["k"][i] * ln["x"][i])
             for i in range(case["n"])]  # fmt: skip
    pl = case["plant"]
    plants = [(_plant_impedance(pl["w_r"][i], pl["rsc_hz"][i], pl["size"][i])
               if pl["has"][i] and (only is None or only == i) else None)
              for i in range(case["n"])]  # fmt: skip
    return lines, plants


def admittance(case: dict, lines: list, plants: list, s: complex) -> np.ndarray:
    n = case["n"]
    y = np.zeros((n, n), dtype=complex)
    for i in range(n):
        yl = 1.0 / lines[i](s)
        y[i, i] += yl
        p = case["parent"][i] - 1  # bus numbers 1..n are rows 0..n-1; parent 0 is the source
        if p >= 0:
            y[p, p] += yl
            y[i, p] -= yl
            y[p, i] -= yl
        if plants[i] is not None:
            y[i, i] += 1.0 / plants[i](s)
    return y


def modes(case: dict, only: int | None = None, iters: int = 40) -> np.ndarray:
    """Sub-synchronous modes as eigenvalues lambda (1/s, rad/s), found by Newton's method
    on det Y(s). Duplicates are removed."""
    lines, plants = impedances(case, only)

    def g(s):
        y = admittance(case, lines, plants, s)
        # Scale each row so the determinant does not over- or underflow.
        return np.linalg.det(y / np.abs(y).max(axis=1, keepdims=True))

    def det(s):
        return np.linalg.det(admittance(case, lines, plants, s))

    s_axis = 1j * SCAN_HZ / F0 + 1e-3
    mag = np.array([abs(g(s)) for s in s_axis])
    starts = [s_axis[k] for k in range(1, len(mag) - 1) if mag[k] <= mag[k - 1] and
              mag[k] <= mag[k + 1]]  # fmt: skip
    found = []
    for s in starts:
        for _ in range(iters):
            h = 1e-7
            f = det(s)
            df = (det(s + h) - det(s - h)) / (2 * h)
            if df == 0 or not np.isfinite(df):
                break
            step = f / df
            s = s - step
            if abs(step) < 1e-12:
                break
        f_er = s.imag * F0
        if 2.0 < f_er < 58.0 and abs(s.real) < 0.5 and abs(det(s)) < 1e-6 * max(1.0, abs(det(
                s + 1e-3))):  # fmt: skip
            lam = 2 * np.pi * F0 * (s - 1j)
            if all(abs(lam - o) > 1e-3 for o in found):
                found.append(lam)
    return np.array(found, dtype=complex)


def growth(case: dict, only: int | None = None) -> tuple[float, float]:
    """Growth rate (1/s) and rotating-frame frequency (Hz) of the least-damped
    sub-synchronous mode; NaN if none is found."""
    lam = modes(case, only)
    if not lam.size:
        return float("nan"), float("nan")
    k = int(np.argmax(lam.real))
    return float(lam.real[k]), float(abs(lam.imag[k]) / (2 * np.pi))


def evaluate(case: dict) -> dict:
    """The truth and the baseline that judges each plant alone: the largest growth rate
    over the plants, each with the others left out."""
    sigma, f = growth(case)
    alone = [growth(case, only=i)[0] for i in range(case["n"]) if case["plant"]["has"][i]]
    alone = [a for a in alone if np.isfinite(a)]
    return {"sigma": sigma, "f": f, "alone_max": max(alone) if alone else float("nan")}


def _job(args) -> tuple[dict, dict]:
    n_bus, seed = args
    case = sample(n_bus, np.random.default_rng(seed))
    return case, evaluate(case)


def generate(sizes: list[int], seed0: int, processes: int = 8) -> list[tuple[dict, dict]]:
    jobs = [(n, SEED + seed0 + k) for k, n in enumerate(sizes)]
    if processes <= 1:
        return [_job(j) for j in jobs]
    from multiprocessing import Pool

    with Pool(processes) as pool:
        return pool.map(_job, jobs, chunksize=32)
