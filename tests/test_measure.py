"""Instruments and mode estimators (step 39)."""

import numpy as np
import pytest

from sso_sciml import measure as me
from sso_sciml.data import t2


def one_mode(sigma: float, f: float, size: float = 0.05) -> np.ndarray:
    """A dataset-shaped record whose line current is one mode in the synchronous frame."""
    t = t2.times()
    obs = np.zeros((1, t2.N_T, 5))
    obs[0, :, 0] = size * np.exp(sigma * t) * np.cos(2 * np.pi * f * t)
    obs[0, :, 1] = 0.6 * size * np.exp(sigma * t) * np.sin(2 * np.pi * f * t + 0.4)
    return obs


def test_phase_current_is_the_fundamental_with_no_deviation():
    cur = me.phase_current(np.zeros((1, t2.N_T, 5)))
    t = me.recorder_times()
    assert np.allclose(cur[0], np.cos(2 * np.pi * me.F0 * t))
    assert t[1] == pytest.approx(1 / 1920) and t[-1] <= t2.T_END


@pytest.mark.parametrize("method", [me.matrix_pencil, me.prony])
def test_estimators_recover_two_exponentials(method):
    t = np.arange(300) * 1e-3
    y = np.exp((-3 + 2j * np.pi * 20) * t) + 0.5 * np.exp((4 - 2j * np.pi * 35) * t)
    s, _ = method(y, 1e-3, 2)
    s = s[np.argsort(s.real)]
    assert np.allclose(s, [-3 + 2j * np.pi * 20, 4 - 2j * np.pi * 35], atol=1e-6)


@pytest.mark.parametrize(("sigma", "f"), [(-4.0, 28.0), (9.0, 41.0)])
def test_recorder_gives_the_mode_in_the_synchronous_frame(sigma, f):
    est = me.from_recorder(me.phase_current(one_mode(sigma, f)), order=6)
    assert est[0] == pytest.approx([sigma, f], abs=0.05)


@pytest.mark.parametrize(("fps", "seen"), [(120, 41.0), (60, 19.0), (30, 11.0)])
def test_phasors_show_the_mode_folded_by_the_frame_rate(fps, seen):
    ph, dt = me.phasors(me.phase_current(one_mode(6.0, 41.0)), fps)
    assert dt == pytest.approx(1 / fps)
    est = me.from_phasors(ph, dt)
    assert est[0, 1] == pytest.approx(seen, abs=0.5)
    assert me.folded(np.array([41.0]), fps)[0] == pytest.approx(seen)
    assert est[0, 0] == pytest.approx(6.0, abs=1.0)


def test_noise_has_the_stated_size():
    rng = np.random.default_rng(0)
    signal = np.ones((3, 4000))
    y = me.add_noise(np.zeros((3, 4000)), signal, 20.0, rng)
    assert y.std() == pytest.approx(0.1, rel=0.05)
    assert np.array_equal(me.add_noise(signal, signal, np.inf, rng), signal)
