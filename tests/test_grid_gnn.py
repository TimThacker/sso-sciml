"""Several plants on a tree network, and the graph network (step 51). CPU."""

import numpy as np
import torch

from sso_sciml.data import grid
from sso_sciml.models import gnn
from sso_sciml.physics import circuit


def test_one_plant_on_one_line_gives_the_root_of_the_loop_impedance():
    for seed in range(4):
        case = grid.sample(1, np.random.default_rng(seed))
        case["plant"]["has"][:] = True
        lines, plants = grid.impedances(case)
        lam = circuit.modes(lines[0] + plants[0])
        f = np.abs(lam.imag) / (2 * np.pi)
        # Sub-synchronous: the electrical resonance below f0, so negative frequency in the
        # synchronous frame (lambda = wb (s - j)).
        band = lam[(lam.imag < 0) & (f > 2) & (f < 58)]
        sigma, _ = grid.growth(case)
        if band.size:
            assert abs(sigma - band.real.max()) < 1e-6
        else:
            assert np.isnan(sigma)


def test_a_plant_on_its_own_is_found_by_leaving_the_others_out():
    case = grid.sample(3, np.random.default_rng(5))
    case["plant"]["has"][:] = True
    lines, plants = grid.impedances(case, only=1)
    assert plants[0] is None and plants[1] is not None and plants[2] is None


def test_graph_network_takes_any_size_and_is_permutation_free_in_the_batch():
    rng = np.random.default_rng(0)
    cases = [grid.sample(n, rng) for n in (1, 3, 6)]
    graphs = [gnn.graph(c) for c in cases]
    torch.manual_seed(0)
    model = gnn.GNN(16, 2)
    with torch.no_grad():
        a = model(*gnn.collate(graphs, "cpu"))
        b = model(*gnn.collate(graphs[::-1], "cpu"))
    assert a.shape == (3,) and np.allclose(a.numpy(), b.numpy()[::-1], atol=1e-6)


def test_graph_network_trains():
    rng = np.random.default_rng(1)
    cases = [grid.sample(int(n), rng) for n in rng.integers(1, 4, 40)]
    graphs = [gnn.graph(c) for c in cases]
    y = np.array([c["line"]["k"].sum() * 10 for c in cases])
    cfg = gnn.Config(steps=60, batch=16, width=16, layers=2, eval_every=30)
    predict, info = gnn.train(graphs[:30], y[:30], (graphs[30:], y[30:]), 0, cfg, "cpu")
    assert predict(graphs).shape == (40,) and len(info["history"]["step"]) == 2
