"""The damping controller, the design problem and the search methods (step 41). CPU."""

import numpy as np
import pytest
import torch

from sso_sciml import design as ds
from sso_sciml import search
from sso_sciml.data import t1
from sso_sciml.solver.linearize import jacobian

CASE = ds.theta_of(0.6, 15.0, ds.SCENARIOS[3])


def test_zero_gain_gives_the_plant_of_dataset_t1():
    out = ds.solve(np.r_[CASE, 0.0, 30.0])
    ref = t1.solve(CASE)
    assert out[5] == 1.0
    assert np.allclose(out[:2], ref[:2], atol=1e-8)


def test_steady_state_is_unchanged_by_the_damping_controller():
    plain, x_plain, _ = t1.operating_point(CASE)
    model, x0, p = ds.operating_point(CASE, 0.8, 25.0)
    assert x0.shape == (29,)
    assert torch.allclose(x0[:25], x_plain)
    assert float(model.rhs(x0, p).abs().max()) < 1e-8
    # The filter output is zero at rest, so the command is that of the plain plant.
    assert float(x0[26].abs() + x0[28].abs()) == 0.0


def test_filter_block_is_a_band_pass_with_unit_gain_at_its_centre():
    model, x0, p = ds.operating_point(CASE, 0.5, 25.0)
    a = jacobian(model.rhs, x0, p).numpy()
    block = a[25:27, 25:27]
    w = 2 * np.pi * 25.0
    assert np.allclose(block, [[0.0, 1.0], [-w * w, -w]])
    # Transfer from the filter's input to z2 at the centre frequency is 1.
    gain = (np.linalg.solve(1j * w * np.eye(2) - block, np.array([0.0, w])))[1]
    assert np.isclose(gain, 1.0)


def test_damping_tuned_to_the_mode_lowers_its_growth_rate():
    base = ds.solve(np.r_[CASE, 0.0, 25.0])
    tuned = ds.solve(np.r_[CASE, 0.6, base[1]])
    off = ds.solve(np.r_[CASE, 0.6, 50.0])
    assert tuned[0] < base[0] - 5.0
    assert off[0] > tuned[0] + 5.0


def test_decode_respects_the_bounds_and_the_integer():
    z = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.39]])
    d = ds.decode(z)
    assert np.allclose(d[0], [ds.FLOOR_HZ, 0.0, 5.0, 1.0])
    assert np.allclose(d[1], [30.0, 0.9, 50.0, ds.MAX_SEGMENTS])
    assert d[2, 3] == 2.0 and np.isclose(d[2, 0], np.sqrt(ds.FLOOR_HZ * 30.0))


def test_score_is_the_worst_scenario_and_failures_count_as_bad():
    out = np.zeros((6, 6))
    out[:, 5] = 1.0
    out[:, 2] = [-3.0, -1.0, -2.0, -5.0, -4.0, -6.0]
    out[4, 5] = 0.0
    assert ds.score(out, 2).tolist() == [-1.0, ds.FAILED]


def test_cases_put_the_segments_into_the_compensation():
    cases = ds.cases_of(np.array([[12.0, 0.4, 20.0, 3.0]]))
    assert cases.shape == (len(ds.SCENARIOS), 14)
    assert np.allclose(cases[:, t1.NAMES.index("k")], 3 * ds.SEGMENT)
    assert np.allclose(cases[:, 12:], [0.4, 20.0])


@pytest.mark.slow
def test_time_response_agrees_in_sign_with_the_eigenvalues():
    stable = np.array([10.0, 0.9, 20.0, 1.0])
    unstable = np.array([15.0, 0.0, 20.0, 5.0])
    sc = ds.SCENARIOS[3]
    assert ds.worst_case(stable, scenarios=sc[None])[0][0] < 0
    assert ds.worst_case(unstable, scenarios=sc[None])[0][0] > 0
    assert ds.time_response(stable, sc, t_end=0.6)["final_over_initial"] < 0.5
    grown = ds.time_response(unstable, sc, t_end=0.6)
    assert (not grown["finite"]) or grown["final_over_initial"] > 2.0


def bowl(z: np.ndarray) -> np.ndarray:
    return ((z - np.array([0.3, 0.7, 0.5])) ** 2).sum(axis=1)


@pytest.mark.parametrize("method", [search.random_search, search.cma_es, search.bayes])
def test_each_method_keeps_its_budget_and_improves(method):
    z, v = method(bowl, 3, 48, 0)
    assert z.shape == (48, 3) and v.shape == (48,)
    assert (z >= 0).all() and (z <= 1).all()
    assert np.allclose(v, bowl(z))
    assert v.min() < 0.03


def test_guided_methods_beat_random_on_a_smooth_function():
    best = {m.__name__: m(bowl, 3, 64, 1)[1].min()
            for m in (search.random_search, search.cma_es, search.bayes)}  # fmt: skip
    assert best["bayes"] < best["random_search"] and best["cma_es"] < best["random_search"]


def test_history_helpers():
    v = np.array([3.0, 4.0, 1.0, 2.0, 0.5])
    assert search.best_so_far(v).tolist() == [3.0, 3.0, 1.0, 1.0, 0.5]
    assert search.calls_to_reach(v, 1.0) == 3 and search.calls_to_reach(v, 0.0) is None
    ei = search.expected_improvement(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 0.5)
    assert ei[0] > ei[1] > 0
