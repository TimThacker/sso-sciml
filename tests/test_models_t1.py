"""Phase 4 checks on the T1 models, run on the committed 1,000-row sample on the CPU."""

from pathlib import Path

import numpy as np
import pytest

from sso_sciml.data import t1 as data
from sso_sciml.models import t1 as mt

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "t1_sample.npz"


@pytest.fixture(scope="module")
def sample():
    z = np.load(SAMPLE)
    ok = z["solver"][:, 4] > 0
    return {"theta": z["theta"][ok], "sigma": z["solver"][ok, 0], "mech": z["mechanism"][ok]}


def test_parameters_scale_to_the_unit_range(sample):
    x = mt.scale_theta(sample["theta"])
    assert x.min() >= -1.0 and x.max() <= 1.0
    assert x.min() < -0.95 and x.max() > 0.95


def test_arm_f_adds_one_input_per_mechanism_column(sample):
    n = len(data.NAMES)
    assert mt.inputs("N", sample["theta"], sample["mech"]).shape[1] == n
    assert mt.inputs("R", sample["theta"], sample["mech"]).shape[1] == n
    x = mt.inputs("F", sample["theta"], sample["mech"])
    assert x.shape[1] == n + len(data.MECHANISM_COLUMNS)
    assert np.isfinite(x).all() and np.abs(x).max() < 10.0


def test_arm_f_can_take_a_subset_of_the_mechanism_columns(sample):
    n = len(data.NAMES)
    x_all = mt.inputs("F", sample["theta"], sample["mech"])
    x = mt.inputs("F", sample["theta"], sample["mech"], ("sigma_impedance", "f_impedance"))
    assert x.shape[1] == n + 2
    j = data.MECHANISM_COLUMNS.index("sigma_impedance")
    assert np.array_equal(x[:, n], x_all[:, n + j])


def test_budget_subset_is_seeded_and_has_no_repeats(sample):
    a = mt.budget_subset(sample, 100, seed=0)
    b = mt.budget_subset(sample, 100, seed=0)
    c = mt.budget_subset(sample, 100, seed=1)
    assert np.array_equal(a["theta"], b["theta"])
    assert not np.array_equal(a["theta"], c["theta"])
    assert len(np.unique(a["theta"], axis=0)) == 100
    assert a["sigma"].shape == (100,)


@pytest.mark.parametrize("arm", mt.ARMS)
def test_targets_invert_to_the_growth_rate(arm, sample):
    y = mt.targets(arm, sample["sigma"], sample["mech"])
    assert np.allclose(mt.to_sigma(arm, y, sample["mech"]), sample["sigma"], rtol=0, atol=1e-12)


def test_arm_r_with_a_zero_correction_is_the_hand_calculation(sample):
    got = mt.to_sigma("R", np.zeros(sample["sigma"].size), sample["mech"])
    assert np.array_equal(got, mt.baseline(sample["mech"]))


def test_metrics_count_screening_errors():
    true = np.array([1.0, 2.0, -1.0, -3.0])
    pred = np.array([-0.5, 2.5, 0.5, -3.0])
    m = mt.metrics(true, pred)
    assert m["n"] == 4
    assert m["mae"] == pytest.approx(0.875)
    assert m["missed_unstable"] == pytest.approx(0.5)
    assert m["false_alarm"] == pytest.approx(0.5)


@pytest.mark.parametrize("arm", mt.ARMS)
def test_training_is_seeded_and_beats_the_mean(arm, sample):
    x = mt.inputs(arm, sample["theta"], sample["mech"])
    y = mt.targets(arm, sample["sigma"], sample["mech"])
    cfg = mt.TrainConfig(steps=300, batch=256, width=64, depth=2, eval_every=100)
    fit, held = slice(0, 800), slice(800, None)
    preds = []
    for _ in range(2):
        model, info = mt.train_mlp(x[fit], y[fit], x[held], y[held], seed=0, cfg=cfg, device="cpu")
        preds.append(mt.predict(model, x[held], device="cpu"))
    assert np.array_equal(preds[0], preds[1])
    sigma = mt.to_sigma(arm, preds[0], sample["mech"][held])
    err = np.abs(sigma - sample["sigma"][held]).mean()
    spread = np.abs(sample["sigma"][held] - sample["sigma"][fit].mean()).mean()
    assert err < 0.5 * spread
    assert len(info["history"]) == 3
