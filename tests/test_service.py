"""The registry and the screening service (step 42). CPU, with a small model built here."""

import copy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sso_sciml import registry, service
from sso_sciml.data import t1, t2
from sso_sciml.models import t1 as mt

# A case inside every training range. Its solver values are pinned below.
CASE = {"k": 0.6, "w_r": 1.0, "rsc_hz": 15.0, "pll_hz": 15.0, "outer_ratio": 0.07, "ff": 1.0,
        "x_line": 1.0, "xr_line": 23.3, "x_t": 0.15, "par_ratio": 8.0, "n_online": 0.4,
        "r_r": 0.00549}  # fmt: skip
GOLDEN = (3.69227, 31.55736)  # growth rate (1/s) and frequency (Hz) from the solver


@pytest.fixture(scope="module")
def bundle():
    theta = t1.sample(640)
    out = t1.generate(theta, 1)
    train, val, test = registry.as_sets(theta, out, t1.folds(640))
    cfg = mt.TrainConfig(steps=300, width=48, depth=3, eval_every=100)
    return registry.build(train, val, "0.0.1", members=3, cfg=cfg, tests={"test": test})


def with_level(bundle, level: float):
    other = copy.copy(bundle)
    other.card = copy.deepcopy(bundle.card)
    other.card["rule"]["level"] = level
    return other


@pytest.fixture(scope="module")
def client(bundle):
    return TestClient(service.create_app(bundle))


# ---------------------------------------------------------------- registry
def test_card_records_what_the_model_was_made_from(bundle):
    card = bundle.card
    assert card["dataset"]["spec_hash"] == t1.spec_hash()
    assert card["dataset"]["version"] == t1.VERSION
    assert len(card["training_commit"]) >= 7
    assert [p["name"] for p in card["inputs"]] == list(t1.NAMES)
    assert all(p["low"] < p["high"] for p in card["inputs"])
    assert {"growth_rate_mae", "solver_share", "missed_unstable"} <= set(card["metrics"]["test"])
    assert card["rule"]["level"] >= 0 and card["rule"]["floor"] > 0


def test_registry_round_trip_keeps_the_predictions(bundle, tmp_path):
    registry.save(bundle, tmp_path)
    assert registry.versions(root=tmp_path) == ["0.0.1"]
    again = registry.load(root=tmp_path)
    theta = t1.sample(8, stream=3)
    mech = np.array([t1.mechanism(row) for row in theta])
    a, b = bundle.predict(theta, mech), again.predict(theta, mech)
    assert np.allclose(a["sigma"], b["sigma"]) and np.allclose(a["f"], b["f"])
    assert again.card["weights_sha256"] == bundle.card["weights_sha256"]


def test_registry_refuses_weights_that_do_not_match_the_card(bundle, tmp_path):
    path = registry.save(bundle, tmp_path)
    blob = bytearray((path / "weights.pt").read_bytes())
    blob[-1] ^= 1
    (path / "weights.pt").write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="do not match"):
        registry.load(root=tmp_path)


def test_registry_without_a_model_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        registry.load(root=tmp_path)


# ---------------------------------------------------------------- contract
def test_health_and_model_card(client, bundle):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model").json()["version"] == bundle.card["version"]


def test_out_of_range_input_is_answered_by_the_solver(client):
    r = client.post("/screen", json={**CASE, "k": 0.9, "w_r": 0.75})
    body = r.json()
    assert r.status_code == 200
    assert body["source"] == "solver" and body["out_of_range"] is True
    assert body["out_of_range_parameters"] == ["k", "w_r"]
    assert body["confidence"] is None
    assert body["interval"]["lower"] == body["interval"]["upper"] == body["growth_rate"]


def test_solver_answer_matches_the_pinned_numbers(bundle):
    out = service.screen(with_level(bundle, float("inf")), np.array(list(CASE.values())))
    assert out.source == "solver" and out.out_of_range is False
    assert out.growth_rate == pytest.approx(GOLDEN[0], abs=1e-3)
    assert out.frequency_hz == pytest.approx(GOLDEN[1], abs=1e-3)
    assert out.stable is False


def test_surrogate_answer_carries_an_interval_and_the_explanation(bundle):
    out = service.screen(with_level(bundle, 0.0), np.array(list(CASE.values())))
    assert out.source == "surrogate" and out.confidence > 0
    assert out.interval.lower < out.growth_rate < out.interval.upper
    assert out.interval.coverage == bundle.card["interval"]["coverage"]
    ex = out.explanation
    assert ex.has_resonance and ex.slip_at_resonance < 0 and ex.r_machine < 0
    assert 0 < ex.resonance_hz < 60
    assert out.hand_calculation.growth_rate == pytest.approx(
        t1.mechanism(np.array(list(CASE.values())))[t1.MECHANISM_COLUMNS.index("sigma_impedance")]
    )


def test_rule_sends_a_prediction_near_zero_to_the_solver(bundle):
    out = service.screen(with_level(bundle, 1e9), np.array(list(CASE.values())))
    assert out.source == "solver" and "too close to zero" in out.reason


def test_bad_input_is_refused(client):
    assert client.post("/screen", json={**CASE, "k": -0.2}).status_code == 422
    missing = {k: v for k, v in CASE.items() if k != "r_r"}
    assert client.post("/screen", json=missing).status_code == 422


# ---------------------------------------------------------------- physics
def test_almost_no_compensation_is_stable(client):
    body = client.post("/screen", json={**CASE, "k": 0.02}).json()
    assert body["source"] == "solver" and body["stable"] is True


def test_growth_rate_rises_with_compensation(client):
    rates = [client.post("/screen", json={**CASE, "k": k, "rsc_hz": 40.0}).json()["growth_rate"]
             for k in (0.2, 0.35, 0.5, 0.65)]  # fmt: skip
    assert all(b > a for a, b in zip(rates[:-1], rates[1:], strict=True))


def test_growth_rate_rises_with_the_current_loop_gain(client):
    rates = [client.post("/screen", json={**CASE, "rsc_hz": hz}).json()["growth_rate"]
             for hz in (35.0, 45.0, 60.0)]  # fmt: skip
    assert all(b > a for a, b in zip(rates[:-1], rates[1:], strict=True))


# ---------------------------------------------------------------- simulate
def test_simulate_returns_a_trajectory(client):
    body = client.post("/simulate", json={"k": 0.1, "w_r": 1.0, "rsc_hz": 10.0, "pll_hz": 10.0})
    assert body.status_code == 200
    d = body.json()
    assert d["quantities"] == list(t2.OBSERVED)
    assert len(d["time_s"]) == t2.N_T and len(d["deviation"]) == t2.N_T
    assert d["deviation"][0][2] == pytest.approx(t2.OFFSET, rel=1e-5)
    assert d["tripped"] is False and d["trip_time_s"] is None
    assert d["growth_rate"] < 0 and d["out_of_range_parameters"] == []
