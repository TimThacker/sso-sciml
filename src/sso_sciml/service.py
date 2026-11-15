"""The screening service (Phase 11).

    POST /screen     growth rate and frequency of the sub-synchronous mode of one case,
                     from the registered model or, when the fallback rule says so, from
                     the solver
    POST /simulate   the response in time of the plant of dataset T2 to a small offset
    GET  /model      the card of the model in use
    GET  /health

`create_app(bundle)` builds the application around a model from `registry`. The rule of
step 40 decides the source of an answer: a parameter outside its training range, or a
prediction too close to zero for its spread, goes to the solver.
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from .data import t1, t2
from .registry import Bundle

_M = {name: j for j, name in enumerate(t1.MECHANISM_COLUMNS)}


def _request_model(name: str, spec) -> type[BaseModel]:
    """One field per parameter. Bounds are wide physical ones, not the training range:
    a case outside the range is answered by the solver, not refused."""
    fields = {p.name: (float, Field(..., gt=0.0 if p.low > 0 else None,
                                    ge=None if p.low > 0 else 0.0,
                                    description=f"{p.meaning}; trained on {p.low:g} to {p.high:g}"))
              for p in spec}  # fmt: skip
    return create_model(name, **fields)


ScreenRequest = _request_model("ScreenRequest", t1.SPEC)
SimulateRequest = _request_model("SimulateRequest", t2.SPEC)


class Interval(BaseModel):
    lower: float
    upper: float
    coverage: float


class Explanation(BaseModel):
    has_resonance: bool
    resonance_hz: float | None
    slip_at_resonance: float | None
    r_machine: float | None
    r_net: float | None
    text: str


class HandCalculation(BaseModel):
    growth_rate: float
    frequency_hz: float
    agrees_on_stability: bool


class ScreenResponse(BaseModel):
    growth_rate: float = Field(description="1/s; positive means the oscillation grows")
    frequency_hz: float = Field(description="frequency of the mode in the rotating frame")
    stable: bool
    interval: Interval
    source: str = Field(description="surrogate or solver")
    reason: str
    out_of_range: bool
    out_of_range_parameters: list[str]
    confidence: float | None
    explanation: Explanation
    hand_calculation: HandCalculation
    model: str


class SimulateResponse(BaseModel):
    time_s: list[float]
    quantities: list[str]
    deviation: list[list[float]] = Field(description="one row per time, one column per quantity")
    tripped: bool
    trip_time_s: float | None
    growth_rate: float
    frequency_hz: float
    out_of_range_parameters: list[str]


def outside(theta: np.ndarray, spec) -> list[str]:
    return [p.name for p, v in zip(spec, theta, strict=True) if not p.low <= v <= p.high]


def explain(mech: np.ndarray) -> Explanation:
    if mech[_M["has_resonance"]] < 0.5:
        return Explanation(has_resonance=False, resonance_hz=None, slip_at_resonance=None,
                           r_machine=None, r_net=None,
                           text="The loop reactance has no zero below synchronous frequency: "
                                "no series resonance for the machine to feed.")  # fmt: skip
    r_m, r_n = float(mech[_M["r_machine"]]), float(mech[_M["r_net"]])
    slip = float(mech[_M["slip_res"]])
    verdict = "negative, so the resonance is fed" if r_n < 0 else "positive, so it is damped"
    return Explanation(
        has_resonance=True, resonance_hz=float(mech[_M["f_er"]]), slip_at_resonance=slip,
        r_machine=r_m, r_net=r_n,
        text=f"At the network resonance the rotor turns faster than the stator field (slip "
             f"{slip:.2f}), so the machine looks like a resistance of {r_m:.4f} pu. With the "
             f"network's resistance the loop has {r_n:.4f} pu: {verdict}.",
    )  # fmt: skip


def screen(bundle: Bundle, theta: np.ndarray) -> ScreenResponse:
    card = bundle.card
    out = outside(theta, t1.SPEC)
    mech = t1.mechanism(theta)
    hand_sigma, hand_f = float(mech[_M["sigma_impedance"]]), float(mech[_M["f_impedance"]])
    source, conf = "solver", None
    reason = "a parameter is outside the range the model was trained on"
    sigma = f = half = None
    if not out:
        p = bundle.predict(theta[None], mech[None])
        sigma, f = float(p["sigma"][0]), float(p["f"][0])
        conf = float(bundle.confidence(p["sigma"], p["spread"])[0])
        half = float(bundle.interval(p["sigma"], p["spread"])[0])
        if conf > card["rule"]["level"]:
            source, reason = "surrogate", "inside the training range and far from the boundary"
        else:
            reason = "the prediction is too close to zero for its spread"
    if source == "solver":
        sol = t1.solve(theta)
        if sol[4] < 0.5:
            raise HTTPException(422, "the solver found no steady state for these parameters")
        sigma, f, half = float(sol[0]), float(sol[1]), 0.0
    return ScreenResponse(
        growth_rate=sigma, frequency_hz=f, stable=sigma <= 0.0,
        interval=Interval(lower=sigma - half, upper=sigma + half,
                          coverage=1.0 if source == "solver" else card["interval"]["coverage"]),
        source=source, reason=reason, out_of_range=bool(out), out_of_range_parameters=out,
        confidence=conf, explanation=explain(mech),
        hand_calculation=HandCalculation(growth_rate=hand_sigma, frequency_hz=hand_f,
                                         agrees_on_stability=(hand_sigma <= 0.0) == (sigma <= 0.0)),
        model=f"{card['name']} {card['version']}",
    )  # fmt: skip


def simulate(theta: np.ndarray) -> SimulateResponse:
    d = t2.generate(theta[None], processes=1, device="cpu")
    if not d["solved"][0]:
        raise HTTPException(422, "the solver found no steady state for these parameters")
    valid = int(d["valid"][0])
    times = t2.times()
    tripped = valid < t2.N_T
    return SimulateResponse(
        time_s=times[:valid].tolist(), quantities=list(t2.OBSERVED),
        deviation=d["observed"][0, :valid].astype(float).tolist(), tripped=tripped,
        trip_time_s=float(times[valid]) if tripped else None,
        growth_rate=float(d["sigma"][0]), frequency_hz=float(d["f"][0]),
        out_of_range_parameters=outside(theta, t2.SPEC),
    )  # fmt: skip


def create_app(bundle: Bundle) -> FastAPI:
    app = FastAPI(title="sso-sciml screening service", version=bundle.card["version"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": f"{bundle.card['name']} {bundle.card['version']}"}

    @app.get("/model")
    def model() -> dict:
        return bundle.card

    @app.post("/screen", response_model=ScreenResponse)
    def post_screen(req: ScreenRequest) -> ScreenResponse:  # type: ignore[valid-type]
        return screen(bundle, np.array([getattr(req, n) for n in t1.NAMES], dtype=float))

    @app.post("/simulate", response_model=SimulateResponse)
    def post_simulate(req: SimulateRequest) -> SimulateResponse:  # type: ignore[valid-type]
        return simulate(np.array([getattr(req, n) for n in t2.NAMES], dtype=float))

    return app
