"""Three ways of minimising an expensive function on the unit cube (Phase 10).

Each takes `f`, which maps an array of points (n, dim) to n values, and a budget of
evaluations, and returns every point tried and its value, in the order tried.

    random_search   scrambled Sobol points
    cma_es          covariance matrix adaptation evolution strategy (the `cma` package)
    bayes           Gaussian process with expected improvement, a few points per round
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy.stats import norm, qmc

Objective = Callable[[np.ndarray], np.ndarray]


def best_so_far(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(values)


def calls_to_reach(values: np.ndarray, target: float) -> int | None:
    """Evaluations used when the value first reaches the target; None if it never does."""
    hit = np.nonzero(np.asarray(values) <= target)[0]
    return int(hit[0]) + 1 if hit.size else None


def random_search(f: Objective, dim: int, evals: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Sobol balance for counts not 2^m
        z = qmc.Sobol(d=dim, scramble=True, seed=seed).random(evals)
    return z, np.asarray(f(z), dtype=float)


def cma_es(f: Objective, dim: int, evals: int, seed: int, popsize: int = 8,
           sigma0: float = 0.3) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    import cma

    es = cma.CMAEvolutionStrategy(
        np.full(dim, 0.5), sigma0,
        {"bounds": [0.0, 1.0], "seed": seed + 1, "popsize": popsize, "verbose": -9},
    )  # fmt: skip
    zs, vs = [], []
    while len(vs) < evals:
        pts = np.clip(np.array(es.ask()), 0.0, 1.0)
        val = np.asarray(f(pts), dtype=float)
        es.tell([p for p in pts], val.tolist())
        zs.append(pts)
        vs.append(val)
    return np.concatenate(zs)[:evals], np.concatenate(vs)[:evals]


def expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    """For minimisation: the expected amount by which a point falls below `best`."""
    sd = np.maximum(sd, 1e-12)
    u = (best - mean) / sd
    return (best - mean) * norm.cdf(u) + sd * norm.pdf(u)


def bayes(f: Objective, dim: int, evals: int, seed: int, initial: int = 16, batch: int = 4,
          candidates: int = 4096,
          cap: float | None = None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Gaussian process (Matern 5/2, one length per coordinate) refitted every round. A
    round proposes `batch` points: the candidate with the largest expected improvement,
    then again with the proposals so far entered at the model's own mean. Values above
    `cap` are entered at `cap`, so that one very bad point does not set the scale."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    rng = np.random.default_rng(seed)
    z, v = random_search(f, dim, initial, seed)
    while len(v) < evals:
        y = np.minimum(v, cap) if cap is not None else v
        kernel = ConstantKernel(1.0) * Matern(np.full(dim, 0.3), (0.02, 5.0), nu=2.5) \
            + WhiteKernel(1e-4, (1e-8, 1e-1))  # fmt: skip
        gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(z, y)
        zz, yy = z.copy(), y.copy()
        new = []
        for _ in range(min(batch, evals - len(v))):
            if new:
                gp_now = GaussianProcessRegressor(gp.kernel_, optimizer=None, normalize_y=True)
                gp_now.fit(zz, yy)
            else:
                gp_now = gp
            # Candidates: uniform, and small moves from the best points so far.
            top = zz[np.argsort(yy)[:8]]
            local = top[rng.integers(0, len(top), candidates // 2)]
            local = np.clip(local + 0.05 * rng.normal(size=local.shape), 0.0, 1.0)
            cand = np.vstack([rng.random((candidates // 2, dim)), local])
            mean, sd = gp_now.predict(cand, return_std=True)
            pick = cand[int(np.argmax(expected_improvement(mean, sd, float(yy.min()))))]
            new.append(pick)
            zz = np.vstack([zz, pick])
            yy = np.r_[yy, gp_now.predict(pick[None])[0]]
        new = np.array(new)
        z, v = np.vstack([z, new]), np.r_[v, np.asarray(f(new), dtype=float)]
    return z[:evals], v[:evals]
