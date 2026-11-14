"""When to trust a learned screen and when to call the solver (Phase 9, task T1).

A detector gives every case a non-negative number that should be large where the model
is wrong. A fallback rule turns the prediction and a detector into one of three answers:
stable, unstable, or "call the solver".

    range_excess    how far the inputs lie outside the training range, in scaled units
    knn_distance    mean distance to the nearest training cases, in any space
    confidence      |prediction| over (detector + floor): large means the sign is safe
    threshold       the confidence below which the solver is called, set on validation
    decide, outcome the rule applied, and its errors and solver share
    tradeoff        errors against the share of cases sent to the solver
    spearman, auroc how well a detector ranks cases by error
"""

from __future__ import annotations

import numpy as np
import torch


def range_excess(u: np.ndarray) -> np.ndarray:
    """Largest distance of any scaled input beyond [-1, 1]; zero inside the range."""
    return np.clip(np.abs(u) - 1.0, 0.0, None).max(axis=1)


def knn_distance(ref: np.ndarray, x: np.ndarray, k: int = 5, device: str = "cpu",
                 exclude_self: bool = False, chunk: int = 2048) -> np.ndarray:  # fmt: skip
    """Mean Euclidean distance from each row of `x` to its `k` nearest rows of `ref`.
    With `exclude_self` the nearest row is dropped, for when `x` is part of `ref`."""
    dev = torch.device(device)
    r = torch.as_tensor(ref, dtype=torch.float32, device=dev)
    out = []
    kk = k + int(exclude_self)
    for lo in range(0, x.shape[0], chunk):
        q = torch.as_tensor(x[lo : lo + chunk], dtype=torch.float32, device=dev)
        near = torch.cdist(q, r).topk(kk, dim=1, largest=False).values
        out.append(near[:, int(exclude_self) :].mean(dim=1).double().cpu().numpy())
    return np.concatenate(out)


def confidence(pred: np.ndarray, detector: np.ndarray, floor: float) -> np.ndarray:
    """Distance of the predicted growth rate from zero, in units of the detector."""
    return np.abs(pred) / (detector + floor)


def threshold(conf: np.ndarray, pred: np.ndarray, sigma: np.ndarray, factor: float = 1.0,
              unstable_only: bool = False) -> float:  # fmt: skip
    """The largest confidence at which a validation case has the wrong sign, times a
    safety factor. Above it no validation case is wrong. With `unstable_only` only
    unstable cases predicted stable count."""
    wrong = (sigma > 0) & (pred <= 0)
    if not unstable_only:
        wrong |= (sigma <= 0) & (pred > 0)
    return float(factor * conf[wrong].max()) if wrong.any() else 0.0


def decide(conf: np.ndarray, pred: np.ndarray, level: float) -> np.ndarray:
    """+1 unstable, -1 stable, 0 call the solver."""
    return np.where(conf > level, np.where(pred > 0, 1, -1), 0)


def outcome(verdict: np.ndarray, sigma: np.ndarray) -> dict:
    """Solver share and the errors among the cases the model answered itself."""
    unstable = sigma > 0
    missed = (verdict == -1) & unstable
    false = (verdict == 1) & ~unstable
    return {
        "solver_share": float((verdict == 0).mean()),
        "missed_unstable": int(missed.sum()),
        "missed_unstable_rate": float(missed.sum() / max(unstable.sum(), 1)),
        "false_unstable": int(false.sum()),
        "false_unstable_rate": float(false.sum() / max((~unstable).sum(), 1)),
        "cases": int(sigma.size),
        "unstable": int(unstable.sum()),
    }


def tradeoff(conf: np.ndarray, pred: np.ndarray, sigma: np.ndarray,
             shares: np.ndarray) -> dict:  # fmt: skip
    """For each share of cases sent to the solver (least confident first): the unstable
    cases called stable and the stable cases called unstable among the rest."""
    order = np.argsort(conf, kind="stable")
    unstable = sigma[order] > 0
    stable_call = pred[order] <= 0
    miss = np.cumsum((unstable & stable_call)[::-1])[::-1]
    false = np.cumsum((~unstable & ~stable_call)[::-1])[::-1]
    n = sigma.size
    cut = np.minimum((np.asarray(shares) * n).round().astype(int), n - 1)
    return {"solver_share": np.asarray(shares).tolist(), "missed_unstable": miss[cut].tolist(),
            "false_unstable": false[cut].tolist()}  # fmt: skip


def share_for_no_miss(conf: np.ndarray, pred: np.ndarray, sigma: np.ndarray) -> float:
    """Smallest share of cases that has to go to the solver, least confident first, for
    no unstable case to be called stable. Uses the true answers: a lower bound."""
    wrong = (sigma > 0) & (pred <= 0)
    if not wrong.any():
        return 0.0
    return float((conf <= conf[wrong].max()).mean())


def _ranks(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a positive case has a higher score than a negative one."""
    positive = positive.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(score.size)
    # Average ranks over ties.
    start = np.r_[0, np.nonzero(np.diff(s))[0] + 1, s.size]
    for lo, hi in zip(start[:-1], start[1:], strict=True):
        ranks[order[lo:hi]] = 0.5 * (lo + hi - 1) + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
