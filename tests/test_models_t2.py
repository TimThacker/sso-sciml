"""Task T2 models (step 36). Small runs on the CPU."""

import numpy as np
import pytest
import torch

from sso_sciml.data import t2
from sso_sciml.models import t2 as mt

CFG = mt.Config(steps=30, batch=256, width=32, depth=2, eval_every=15)


@pytest.fixture(scope="module")
def cases():
    d = t2.generate(t2.sample(8), processes=1, device="cpu")
    assert d["solved"].all()
    return {k: v for k, v in d.items() if k != "solved"}


@pytest.fixture(scope="module")
def pool():
    return mt.unlabelled(t2.sample(8)[:4])


def test_observation_matrix_gives_the_observed_quantities(cases):
    got = cases["states"] @ t2.observation_matrix().T.astype(np.float32)
    assert np.abs(got - cases["observed"]).max() < 2e-4


def test_error_of_the_truth_is_zero_and_of_nothing_is_one(cases):
    assert mt.case_error(mt.targets(cases), cases) == pytest.approx(0.0, abs=1e-7)
    assert mt.case_error(np.zeros_like(cases["observed"]), cases) == pytest.approx(1.0)


def test_arm_a_before_training_is_the_reduced_model(cases):
    torch.manual_seed(0)
    model = mt.LinearSystem(4, 32, 2)
    with torch.no_grad():
        model.net[-1].weight.zero_()
        out = model(torch.zeros(8, 4), torch.as_tensor(cases["a_red"], dtype=torch.float32))
    assert np.abs(out.numpy() - mt.reduced_alone(cases)).max() < 0.02


def test_arm_an_before_training_is_arm_a(cases):
    a_red = torch.as_tensor(cases["a_red"], dtype=torch.float32)
    theta = torch.randn(8, 4)
    torch.manual_seed(0)
    lin = mt.LinearSystem(4, 32, 2)
    torch.manual_seed(0)
    node = mt.LinearSystem(4, 32, 2, nonlinear=True)
    with torch.no_grad():
        assert torch.allclose(lin(theta, a_red), node(theta, a_red))
        node.g[-1].bias.fill_(0.3)  # a constant term is taken off: the steady state holds
        still = node.z0.clone()
        node.z0.zero_()
        assert node(theta, a_red).abs().max() < 1e-6
        node.z0.copy_(still)


@pytest.mark.parametrize("arm", mt.ARMS)
def test_every_arm_trains_and_starts_from_the_known_initial_state(arm, cases, pool):
    lab = {k: v[:6] for k, v in cases.items()}
    held = {k: v[6:] for k, v in cases.items()}
    predict, info = mt.train(arm, lab, held, 0, CFG, pool=pool, device="cpu")
    pred = predict(cases)
    assert pred.shape == cases["observed"].shape and np.isfinite(pred).all()
    assert np.allclose(pred[:, 0], [0.0, 0.0, 1.0, 0.0, 0.0], atol=2e-3)
    assert len(info["history"]["step"]) == 2 and info["skipped_steps"] == 0
    if arm in ("L", "FL"):
        assert np.isfinite(info["history"]["residual"]).all()


def test_training_is_fixed_by_the_seed(cases):
    lab = {k: v[:6] for k, v in cases.items()}
    held = {k: v[6:] for k, v in cases.items()}
    a = mt.train("N", lab, held, 3, CFG, device="cpu")[0](cases)
    b = mt.train("N", lab, held, 3, CFG, device="cpu")[0](cases)
    assert np.array_equal(a, b)
