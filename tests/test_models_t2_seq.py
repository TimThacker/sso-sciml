"""Operators and sequence models on task T2 (step 38). Small runs on the CPU."""

import numpy as np
import pytest

from sso_sciml.data import t2
from sso_sciml.models import t2 as mt
from sso_sciml.models import t2_seq as ms

CFG = ms.Config(steps=20, cases=4, width=16, eval_every=10)


@pytest.fixture(scope="module")
def cases():
    d = t2.generate(t2.sample(8), processes=1, device="cpu")
    assert d["solved"].all()
    return {k: v for k, v in d.items() if k != "solved"}


@pytest.mark.parametrize("arch", ms.ARCHS)
@pytest.mark.parametrize("features", [False, True])
def test_every_family_trains_and_starts_from_the_known_initial_state(arch, features, cases):
    lab = {k: v[:6] for k, v in cases.items()}
    held = {k: v[6:] for k, v in cases.items()}
    predict, info = ms.train(arch, features, lab, held, 0, CFG, device="cpu")
    pred = predict(cases)
    assert pred.shape == cases["observed"].shape and np.isfinite(pred).all()
    assert np.allclose(pred[:, 0], [0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6)
    assert len(info["history"]["step"]) == 2 and info["skipped_steps"] == 0
    assert info["parameters"] > 0


def test_training_is_fixed_by_the_seed(cases):
    lab = {k: v[:6] for k, v in cases.items()}
    held = {k: v[6:] for k, v in cases.items()}
    a = ms.train("tcn", True, lab, held, 3, CFG, device="cpu")[0](cases)
    b = ms.train("tcn", True, lab, held, 3, CFG, device="cpu")[0](cases)
    assert np.array_equal(a, b)


def test_horizon_error_of_the_truth_is_zero_and_of_nothing_is_one(cases):
    edges = np.array([0, 67, 134, 201])
    whole = cases["valid"] >= t2.N_T
    assert np.nanmax(ms.horizon_error(mt.targets(cases), cases, edges)) == pytest.approx(0.0)
    nothing = ms.horizon_error(np.zeros_like(cases["observed"]), cases, edges)
    assert np.allclose(nothing[whole], 1.0)


def test_growth_reads_the_rate_of_an_exponential():
    t = t2.times()
    y = np.stack([np.exp(s * t)[:, None] * np.ones((1, 5)) for s in (-8.0, 0.0, 12.0)])
    assert np.allclose(ms.growth(y), [-8.0, 0.0, 12.0], atol=1e-6)


def test_growth_scores_of_the_truth(cases):
    s = ms.growth_scores(mt.targets(cases), cases)
    assert s["growth_error_median"] == 0.0
    assert not s["false_divergence"] > 0 and not s["missed_growth"] > 0
