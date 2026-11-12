"""The slow mode of the doubly-fed plant and the reduced models asked to reproduce it."""

import numpy as np
import pytest

from sso_sciml import slowmode
from sso_sciml.data import t1

COL = {name: j for j, name in enumerate(slowmode.COLUMNS)}


@pytest.fixture(scope="module")
def rows():
    return slowmode.cases(t1.sample(8))


def test_row_matches_its_columns(rows):
    assert rows.shape == (8, len(slowmode.COLUMNS))
    assert np.isfinite(rows[:, COL["true_slow"]]).all()


def test_participation_sums_to_one(rows):
    part = rows[:, [COL[f"part_{k}"] for k in range(slowmode.N_STATE)]]
    assert np.allclose(part.sum(axis=1), 1.0)


def test_phasor_model_keeps_the_slow_mode_and_loses_the_resonance(rows):
    assert np.abs(rows[:, COL["phasor_slow"]] - rows[:, COL["true_slow"]]).max() < 0.05
    # The full model has a mode between 2 and 58 Hz in every case; the phasor-domain
    # model's modes there, if any, are not that one.
    assert np.isfinite(rows[:, COL["true_sub"]]).all()
    assert not np.allclose(rows[:, COL["phasor_sub"]], rows[:, COL["true_sub"]], atol=0.5)


def test_slow_mode_is_slow(rows):
    assert (rows[:, COL["f_slow"]] < 2.0).all()


def test_unstable_slow_case_needs_the_network():
    # Slow current loop, fast outer loops, top of the speed range: unstable on its
    # network, stable alone on an ideal voltage, and the phasor-domain model agrees.
    theta = np.array([0.435, 1.19, 2.04, 8.88, 0.108, 0.28, 1.45, 17.4, 0.173, 0.833, 0.974,
                      0.00526])  # fmt: skip
    row = slowmode.case(theta)
    assert row[COL["true_slow"]] > 0.1
    assert row[COL["alone_slow"]] < 0
    assert row[COL["phasor_slow"]] == pytest.approx(row[COL["true_slow"]], abs=0.02)
