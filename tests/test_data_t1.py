"""Phase 3 checks on the T1 dataset.

The committed sample in data/samples/ is the first 1,000 in-distribution rows of the
full dataset. These tests fail if the solver, the hand calculation, the ranges or the
sampling change.
"""

from pathlib import Path

import numpy as np
import pytest

from sso_sciml.data import t1

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "t1_sample.npz"


@pytest.fixture(scope="module")
def sample():
    return np.load(SAMPLE)


def test_spec_is_the_one_the_dataset_was_built_with():
    assert t1.VERSION == "t1-v2"
    assert t1.spec_hash() == "bb4ab735373a3947"
    assert len(t1.NAMES) == 12


def test_sampling_is_reproducible_and_inside_the_ranges(sample):
    theta = t1.sample(131072)[:1000]
    # The Sobol points are exact on every platform. Parameters sampled in log pass
    # through exp and log, whose last bit differs between the Windows and Linux builds
    # of NumPy, so those columns are compared to rounding error only.
    for j, par in enumerate(t1.SPEC):
        if par.log:
            assert np.allclose(theta[:, j], sample["theta"][:, j], rtol=1e-13, atol=0)
        else:
            assert np.array_equal(theta[:, j], sample["theta"][:, j])
    for j, par in enumerate(t1.SPEC):
        assert theta[:, j].min() >= par.low and theta[:, j].max() <= par.high


def test_extrapolation_splits_leave_the_range_in_one_parameter_only():
    for split, moved in t1.SPLITS.items():
        theta = t1.sample(64, split)
        for j, par in enumerate(t1.SPEC):
            lo, hi = moved.get(par.name, (par.low, par.high))
            assert theta[:, j].min() >= lo and theta[:, j].max() <= hi
        for name, (lo, _) in moved.items():
            par = t1.SPEC[t1.NAMES.index(name)]
            assert lo >= par.high or moved[name][1] <= par.low


def test_solver_and_mechanism_columns_regenerate(sample):
    rows = [0, 1, 2, 3, 500, 999]
    got = np.array([t1.evaluate(sample["theta"][r]) for r in rows])
    want = np.hstack([sample["solver"][rows], sample["mechanism"][rows]])
    assert np.allclose(got, want, rtol=0, atol=1e-9, equal_nan=True)


def test_folds_are_fixed():
    fold = t1.folds(131072)
    assert [(fold == k).sum() for k in range(3)] == [104857, 13107, 13108]
    assert fold[:8].tolist() == t1.folds(131072)[:8].tolist()


def test_hand_calculation_carries_most_of_the_answer_but_not_all(sample):
    solver, mech = sample["solver"], sample["mechanism"]
    ok = (solver[:, 4] > 0) & (mech[:, 0] > 0)
    err_circuit = np.abs(mech[ok, 6] - solver[ok, 0]).mean()
    spread = np.abs(solver[ok, 0] - solver[ok, 0].mean()).mean()
    assert err_circuit < 0.5 * spread  # useful as a feature
    assert err_circuit > 0.5  # and still leaves something to learn, in 1/s


def test_boundary_search_lands_on_the_boundary():
    theta = t1.sample(16, stream=1)
    found = t1.boundary_cases(theta)
    assert len(found) >= 1
    sigma = np.array([t1.solve(row)[0] for row in found])
    assert np.abs(sigma).max() < 1.0
