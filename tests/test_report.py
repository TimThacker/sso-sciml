"""The shared plots draw without error and show what they claim to."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sso_sciml import report


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_curves_draw_each_run_and_their_mean(ax):
    x = np.arange(5.0)
    report.curves(ax, [(x, x), (x, x + 2.0)], "#2a78d6", "arm")
    assert len(ax.lines) == 3
    assert np.allclose(ax.lines[-1].get_ydata(), x + 1.0)
    assert ax.lines[-1].get_label() == "arm"


def test_parity_states_the_mean_error(ax):
    true = np.linspace(-5.0, 5.0, 200)
    report.parity(ax, true, true + 0.5, (-6.0, 6.0))
    assert any("0.50" in t.get_text() for t in ax.texts)
    assert ax.get_xlim() == (-6.0, 6.0)


def test_error_cdf_ends_at_one_and_is_sorted(ax):
    report.error_cdf(ax, {"a": np.array([0.3, -0.1, 0.2])}, {"a": "#2a78d6"})
    x, y = ax.lines[0].get_data()
    assert np.allclose(x, [0.1, 0.2, 0.3]) and y[-1] == 1.0


def test_binned_error_skips_thin_bins(ax):
    x = np.concatenate([np.full(30, 0.5), np.full(5, 1.5)])
    report.binned_error(ax, x, np.ones_like(x), np.array([0.0, 1.0, 2.0]), "#2a78d6", "a")
    xs, ys = ax.lines[0].get_data()
    assert list(xs) == [0.5] and list(ys) == [1.0]


def test_reliability_draws_the_diagonal_and_each_method(ax):
    levels = np.array([0.5, 0.9])
    report.reliability(ax, levels, {"m": np.array([0.4, 0.8])}, {"m": "#2a78d6"})
    assert len(ax.lines) == 2
    assert np.allclose(ax.lines[1].get_ydata(), [0.4, 0.8])
