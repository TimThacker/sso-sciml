"""Paired statistics and the model-as-a-screen score of Phase 6."""

import numpy as np
import pytest

from sso_sciml.models import compare as cmp


def test_safe_margin_clears_no_unstable_case():
    sigma = np.array([-3.0, -1.0, 0.5, 2.0, -0.2])
    pred = np.array([-2.5, -1.5, -0.7, 1.0, -0.1])
    margin = cmp.safe_margin(pred, sigma)
    assert margin == pytest.approx(0.7)
    s = cmp.scores(sigma, pred, margin)
    assert s["unstable_cleared"] == 0
    assert s["cleared"] == pytest.approx(2 / 5)
    # With no margin the unstable case predicted at -0.7 is cleared.
    assert cmp.scores(sigma, pred, 0.0)["unstable_cleared"] == 1
    assert cmp.safe_margin(np.array([1.0, -1.0]), np.array([2.0, -2.0])) == 0.0


def test_interval_is_the_t_interval_of_five_differences():
    from scipy import stats

    v = np.array([0.1, 0.3, 0.2, 0.25, 0.05])
    got = cmp.interval(v)
    lo, hi = stats.t.interval(0.95, 4, loc=v.mean(), scale=stats.sem(v))
    assert (got["low"], got["high"]) == pytest.approx((lo, hi))
    assert got["differs"]
    assert not cmp.interval(np.array([0.1, -0.3, 0.2, -0.25, 0.05]))["differs"]


def test_change_is_relative_and_paired():
    b = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    got = cmp.change(0.5 * b, b)
    assert got["mean"] == pytest.approx(-0.5) and got["low"] == pytest.approx(-0.5)
    assert got["differs"]
