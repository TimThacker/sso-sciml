"""Detectors and the fallback rule (step 40). CPU."""

import numpy as np

from sso_sciml import fallback as fb


def test_range_excess_is_zero_inside_and_the_largest_overshoot_outside():
    u = np.array([[0.2, -0.9, 1.0], [0.0, -1.5, 1.2], [2.0, 0.0, 0.0]])
    assert np.allclose(fb.range_excess(u), [0.0, 0.5, 1.0])


def test_knn_distance_matches_a_direct_calculation():
    rng = np.random.default_rng(0)
    ref, x = rng.normal(size=(200, 3)), rng.normal(size=(17, 3))
    d = np.sort(np.linalg.norm(x[:, None] - ref[None], axis=2), axis=1)
    assert np.allclose(fb.knn_distance(ref, x, k=4, chunk=5), d[:, :4].mean(axis=1), atol=1e-5)
    own = np.sort(np.linalg.norm(ref[:, None] - ref[None], axis=2), axis=1)[:, 1:4].mean(axis=1)
    assert np.allclose(fb.knn_distance(ref, ref, k=3, exclude_self=True), own, atol=1e-4)


def test_threshold_leaves_no_wrong_sign_above_it():
    rng = np.random.default_rng(1)
    sigma = rng.normal(size=2000)
    pred = sigma + 0.3 * rng.normal(size=2000)
    conf = fb.confidence(pred, np.full(2000, 0.1), 0.0)
    level = fb.threshold(conf, pred, sigma)
    out = fb.outcome(fb.decide(conf, pred, level), sigma)
    assert out["missed_unstable"] == 0 and out["false_unstable"] == 0
    assert 0.0 < out["solver_share"] < 1.0
    only = fb.threshold(conf, pred, sigma, unstable_only=True)
    assert only <= level
    assert fb.outcome(fb.decide(conf, pred, only), sigma)["missed_unstable"] == 0


def test_decide_gives_three_answers():
    v = fb.decide(np.array([0.5, 2.0, 2.0]), np.array([1.0, 1.0, -1.0]), 1.0)
    assert v.tolist() == [0, 1, -1]


def test_tradeoff_counts_fall_with_the_solver_share_and_agree_with_outcome():
    rng = np.random.default_rng(2)
    sigma = rng.normal(size=3000)
    pred = sigma + 0.5 * rng.normal(size=3000)
    conf = np.abs(pred)
    t = fb.tradeoff(conf, pred, sigma, np.array([0.0, 0.1, 0.3, 0.6]))
    assert all(np.diff(t["missed_unstable"]) <= 0) and all(np.diff(t["false_unstable"]) <= 0)
    assert t["missed_unstable"][0] == int(((sigma > 0) & (pred <= 0)).sum())
    level = np.sort(conf)[299]
    assert fb.outcome(fb.decide(conf, pred, level), sigma)["missed_unstable"] == (
        t["missed_unstable"][1]
    )


def test_share_for_no_miss_is_the_smallest_that_works():
    sigma = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    pred = np.array([-0.2, 3.0, -4.0, -5.0, 6.0])
    conf = np.abs(pred)
    assert fb.share_for_no_miss(conf, pred, sigma) == 0.2
    assert fb.share_for_no_miss(conf, np.abs(pred) * np.sign(sigma), sigma) == 0.0


def test_rank_measures():
    a = np.arange(50.0)
    assert np.isclose(fb.spearman(a, a**3), 1.0) and np.isclose(fb.spearman(a, -a), -1.0)
    assert fb.spearman(a, np.ones(50)) == 0.0
    assert fb.auroc(a, a >= 25) == 1.0 and fb.auroc(-a, a >= 25) == 0.0
    assert np.isclose(fb.auroc(np.ones(50), a >= 25), 0.5)
