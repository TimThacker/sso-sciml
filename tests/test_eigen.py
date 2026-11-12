"""Phase 5 checks on the eigen-residual model (arm L), on the CPU."""

from pathlib import Path

import numpy as np
import pytest
import torch

from sso_sciml.data import t1 as data
from sso_sciml.models import eigen
from sso_sciml.models import t1 as mt

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "t1_sample.npz"


def test_residual_is_zero_at_a_true_eigenpair_and_not_elsewhere():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 6, 6)) * 50.0
    lam, vec = np.linalg.eig(a)
    k = np.argmax(lam.imag, axis=1)
    lam = lam[np.arange(3), k]
    v = vec[np.arange(3), :, k]
    f = lam.imag / (2 * np.pi)
    assert eigen.residual_norm(a, lam.real, f, v).max() < 1e-9
    assert eigen.residual_norm(a, lam.real + 1.0, f, v).min() == pytest.approx(1.0)
    got = eigen.eigen_residual(
        torch.as_tensor(a / eigen.RATE),
        torch.as_tensor(lam.real / mt.SIGMA_SCALE),
        torch.as_tensor(f / eigen.F_SCALE),
        torch.as_tensor(v.real),
        torch.as_tensor(v.imag),
    )
    assert float(got.max()) < 1e-18
    args = (torch.as_tensor(a / eigen.RATE), torch.as_tensor(lam.real / mt.SIGMA_SCALE),
            torch.as_tensor(f / eigen.F_SCALE))  # fmt: skip
    assert float(eigen.singular_residual(*args).max()) < 1e-18
    shifted = eigen.singular_residual(args[0], args[1] + 1.0 / mt.SIGMA_SCALE, args[2])
    assert float(shifted.min()) > 0 and float(shifted.max()) <= (1.0 / eigen.RATE) ** 2 + 1e-12


def test_singular_residual_pulls_a_near_guess_onto_the_eigenvalue():
    a = np.array([[-2.0, -80.0], [80.0, -2.0]])  # eigenvalues -2 +/- 80j
    x = np.zeros((64, 1))
    lab = {"x": x, "sigma": np.full(64, 1.0), "f": np.full(64, 14.0)}  # labels off by 3 and 1.3
    pool = {"x": x, "a": np.repeat(a[None], 64, axis=0)}
    cfg = eigen.EigenConfig(steps=1500, batch=64, width=32, depth=2, eval_every=500, weight=1e4,
                            form="singular")  # fmt: skip
    model, _ = eigen.train_eigen(lab, None, pool, seed=0, cfg=cfg, device="cpu")
    out = eigen.predict(model, x[:1], device="cpu")
    assert abs(out["sigma"][0] + 2.0) < 0.3
    # Either member of the conjugate pair is an eigenvalue; a strong penalty may pick either.
    assert abs(abs(out["f"][0]) - 80.0 / (2 * np.pi)) < 0.1


def test_network_returns_a_unit_eigenvector():
    torch.manual_seed(0)
    net = eigen.EigenNet(4, 5, width=16, depth=2)
    _, _, v_r, v_i = net(torch.randn(7, 4))
    assert torch.allclose((v_r**2 + v_i**2).sum(dim=1), torch.ones(7), atol=1e-6)


def test_state_matrix_carries_the_sub_synchronous_mode_of_the_dataset():
    z = np.load(SAMPLE)
    for row in (0, 1, 500):
        a = data.state_matrix(z["theta"][row])
        lam = np.linalg.eigvals(a)
        target = z["solver"][row, 0] + 2j * np.pi * z["solver"][row, 1]
        assert np.abs(lam - target).min() < 1e-6


def test_residual_alone_finds_an_eigenpair_of_a_fixed_matrix():
    a = np.array([[-2.0, -80.0], [80.0, -2.0]])  # eigenvalues -2 +/- 80j
    pool = {"x": np.zeros((64, 1)), "a": np.repeat(a[None], 64, axis=0)}
    cfg = eigen.EigenConfig(steps=1500, batch=64, width=32, depth=2, eval_every=500)
    model, info = eigen.train_eigen(None, None, pool, seed=0, cfg=cfg, device="cpu")
    out = eigen.predict(model, pool["x"][:1], device="cpu")
    assert abs(out["sigma"][0] + 2.0) < 0.5
    assert abs(abs(out["f"][0]) - 80.0 / (2 * np.pi)) < 0.2
    assert info["history"]["residual"][-1] < info["history"]["residual"][0]
    assert eigen.residual_norm(pool["a"][:1], out["sigma"], out["f"], out["v"])[0] < 1.0


def test_data_loss_alone_fits_labels_and_keeps_the_best_checkpoint():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (400, 2))
    lab = {"x": x[:300], "sigma": 5.0 * x[:300, 0], "f": 20.0 + 5.0 * x[:300, 1]}
    held = {"x": x[300:], "sigma": 5.0 * x[300:, 0], "f": 20.0 + 5.0 * x[300:, 1]}
    cfg = eigen.EigenConfig(steps=600, batch=128, width=32, depth=2, eval_every=100, weight=0.0)
    model, info = eigen.train_eigen(lab, held, None, seed=0, cfg=cfg, device="cpu")
    out = eigen.predict(model, held["x"], device="cpu")
    assert np.abs(out["sigma"] - held["sigma"]).mean() < 0.3
    assert np.abs(out["f"] - held["f"]).mean() < 0.5
    h = info["history"]
    assert np.abs(out["sigma"] - held["sigma"]).mean() == pytest.approx(
        min(h["held_out_mae"]), abs=1e-5
    )
