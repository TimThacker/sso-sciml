"""Environment checks. The GPU test is skipped on machines without CUDA (for example CI)."""

import pytest
import torch

import sso_sciml


def test_package_imports():
    assert sso_sciml.__version__


def test_float64_eigvals_matches_known_spectrum():
    # Companion matrix of (s + 1)(s + 2)(s^2 + 2s + 5): roots -1, -2, -1 +/- 2j.
    a = torch.tensor(
        [
            [-5.0, -13.0, -19.0, -10.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    eig = torch.linalg.eigvals(a)
    got = sorted(eig.tolist(), key=lambda z: (round(z.real, 9), round(z.imag, 9)))
    want = [complex(-2, 0), complex(-1, -2), complex(-1, 0), complex(-1, 2)]
    for g, w in zip(got, want, strict=True):
        assert abs(g - w) < 1e-9


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(256, 64, generator=gen, dtype=torch.float64)
    cpu = x @ x.T
    gpu = (x.cuda() @ x.cuda().T).cpu()
    assert torch.allclose(cpu, gpu, rtol=1e-12, atol=1e-12)
