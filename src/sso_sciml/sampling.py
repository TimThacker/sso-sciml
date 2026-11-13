"""Taking part of a dataset.

The datasets are scrambled Sobol sequences. Their rows are not in random order: the low
bits of the row index set the high bits of every coordinate, so `theta[::4]` is not a
quarter of the dataset but one corner of it. For the converters of dataset C1 with a
phase-locked loop it keeps only positive power and short-circuit ratios above 3.46; for
those without, only ratios below 3.46 and loads under 0.55 pu (tests/test_sampling.py).
Steps 20, 23, 24 and 26 first did exactly that and were rerun (see
docs/steps/29-restated-with-delay.md).

Use `part` wherever fewer rows than all are wanted.
"""

from __future__ import annotations

import numpy as np


def part(n: int, every: int, seed: int = 20261112) -> np.ndarray:
    """Indices of n // every rows chosen at random without replacement, in rising order.
    The same (n, every, seed) always returns the same rows. `every` = 1 returns all."""
    if every <= 1:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).permutation(n)[: n // every])
