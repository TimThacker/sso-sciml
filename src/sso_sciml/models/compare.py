"""Shared by the Phase 6 scripts: data loading, a model used as a screen, and paired
statistics over seeds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data import t1 as data
from . import t1 as mt

TESTS = ("in_distribution", "boundary", *list(data.SPLITS)[1:])
BUDGETS = (100, 1000, 10000, 100000)
SEEDS = (0, 1, 2, 3, 4)
T_975_4 = 2.7764451051977987  # 97.5th percentile of Student t with 4 degrees of freedom


def load(root: Path, split: str) -> dict:
    z = np.load(root / f"{split}.npz")
    ok = z["solver"][:, 4] > 0
    out = {"theta": z["theta"][ok], "sigma": z["solver"][ok, 0], "f": z["solver"][ok, 1],
           "mech": z["mechanism"][ok]}  # fmt: skip
    if "fold" in z:
        out["fold"] = z["fold"][ok]
    return out


def take(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def sets(root: Path) -> tuple[dict, dict, dict]:
    """Training fold, validation fold and the test sets of dataset T1."""
    full = load(root, "in_distribution")
    tests = {"in_distribution": take(full, full["fold"] == 2), "boundary": load(root, "boundary")}
    tests.update({s: load(root, s) for s in TESTS[2:]})
    return take(full, full["fold"] == 0), take(full, full["fold"] == 1), tests


def safe_margin(pred: np.ndarray, sigma: np.ndarray) -> float:
    """Smallest margin (1/s) at which no unstable case is predicted below minus it."""
    unstable = sigma > 0
    return float(max(0.0, -pred[unstable].min())) if unstable.any() else 0.0


def scores(sigma: np.ndarray, pred: np.ndarray, margin: float) -> dict:
    """The metrics of step 09, and the model as a screen: the share of cases predicted
    below minus the margin, and the unstable cases among them."""
    out = mt.metrics(sigma, pred)
    cleared = pred < -margin
    out["cleared"] = float(cleared.mean())
    out["unstable_cleared"] = int((cleared & (sigma > 0)).sum())
    return out


def interval(values: np.ndarray) -> dict:
    """Mean of five paired differences with its 95 % interval."""
    values = np.asarray(values, dtype=float)
    assert values.size == 5, "the t quantile is for five seeds"
    mean, half = float(values.mean()), float(T_975_4 * values.std(ddof=1) / np.sqrt(values.size))
    return {"mean": mean, "low": mean - half, "high": mean + half,
            "differs": bool(abs(mean) > half)}  # fmt: skip


def change(a: np.ndarray, b: np.ndarray) -> dict:
    """Relative change of a positive quantity from b to a, paired, from the log ratio."""
    out = interval(np.log(np.asarray(a, dtype=float) / np.asarray(b, dtype=float)))
    return {k: (float(np.expm1(v)) if k != "differs" else v) for k, v in out.items()}
