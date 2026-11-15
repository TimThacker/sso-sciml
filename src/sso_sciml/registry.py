"""Model registry for the growth-rate screen (Phase 11).

A registered model is a directory `<root>/<name>/<version>/` with two files:

    weights.pt   the network weights
    card.json    version, dataset hash, training commit, metrics, valid input ranges,
                 and the constants of the fallback rule

The model is the screen of step 40: arm R of task T1 as an ensemble (the hand
calculation's growth rate plus a learned correction), with a second network for the
frequency of the mode. Weights are not committed to the repository; `build` makes them
from the dataset, and `card.json` records what they were made from.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from . import fallback as fb
from .data import t1
from .models import t1 as mt

NAME = "t1-screen"
ROOT = Path(os.environ.get("SSO_REGISTRY", Path(__file__).resolve().parents[2] / "registry"))
F_SCALE = 10.0  # Hz
_F = t1.MECHANISM_COLUMNS.index("f_impedance")


@dataclass
class Bundle:
    card: dict
    members: list[mt.MLP]
    freq: mt.MLP
    device: str = "cpu"

    def predict(self, theta: np.ndarray, mech: np.ndarray) -> dict:
        """Growth rate (mean of the members), their spread, and the frequency."""
        x = mt.inputs("R", theta, mech)
        p = np.array([mt.to_sigma("R", mt.predict(m, x, self.device), mech) for m in self.members])
        f = mt.predict(self.freq, x, self.device) * F_SCALE + mech[:, _F]
        return {"sigma": p.mean(axis=0), "spread": p.std(axis=0), "f": f}

    def confidence(self, sigma: np.ndarray, spread: np.ndarray) -> np.ndarray:
        return fb.confidence(sigma, spread, self.card["rule"]["floor"])

    def interval(self, sigma: np.ndarray, spread: np.ndarray) -> np.ndarray:
        """Half-width of the interval of the growth rate at the card's coverage."""
        return self.card["interval"]["quantile"] * (spread + self.card["rule"]["floor"])


def commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def input_ranges() -> list[dict]:
    return [{"name": p.name, "low": p.low, "high": p.high, "log": p.log, "meaning": p.meaning}
            for p in t1.SPEC]  # fmt: skip


def as_sets(theta: np.ndarray, out: np.ndarray, fold: np.ndarray) -> tuple[dict, dict, dict]:
    """Training, validation and test dictionaries from the rows `t1.generate` returns."""
    n_solver = len(t1.SOLVER_COLUMNS)
    ok = out[:, 4] > 0
    full = {"theta": theta[ok], "sigma": out[ok, 0], "f": out[ok, 1], "mech": out[ok, n_solver:]}
    return tuple({k: v[fold[ok] == j] for k, v in full.items()} for j in range(3))


def dataset(data_dir: Path, quick_cases: int = 16384, processes: int = 8) -> tuple:
    """The sets of dataset T1 from `data_dir` if it is there (see
    scripts/phase3_generate_t1.py). If not, `quick_cases` fresh cases from the solver,
    which takes about a minute. Returns the three sets and where they came from."""
    path = data_dir / "in_distribution.npz"
    if path.is_file():
        z = np.load(path)
        out = np.hstack([z["solver"], z["mechanism"]])
        return (*as_sets(z["theta"], out, z["fold"]), f"{t1.VERSION}, {len(out)} cases")
    theta = t1.sample(quick_cases)
    out = t1.generate(theta, processes)
    return (*as_sets(theta, out, t1.folds(quick_cases)),
            f"{quick_cases} cases generated for this build")  # fmt: skip


def build(train: dict, val: dict, version: str, members: int = 5, factor: float = 1.5,
          coverage: float = 0.9, cfg: mt.TrainConfig | None = None, device: str = "cpu",
          tests: dict | None = None) -> Bundle:  # fmt: skip
    """Train the ensemble and the frequency network on `train`, set the interval on the
    first half of `val` and the rule on the second half, and score on `tests`. Each of
    the three is a dictionary with `theta`, `sigma`, `f` and `mech`."""
    x = mt.inputs("R", train["theta"], train["mech"])
    y = mt.targets("R", train["sigma"], train["mech"])
    h = max(len(x) // 5, 1)
    nets = [mt.train_mlp(x[h:], y[h:], x[:h], y[:h], seed, cfg, device)[0]
            for seed in range(members)]  # fmt: skip
    yf = (train["f"] - train["mech"][:, _F]) / F_SCALE
    freq = mt.train_mlp(x[h:], yf[h:], x[:h], yf[:h], 100, cfg, device)[0]
    bundle = Bundle({"rule": {"floor": 0.0}, "interval": {"quantile": 0.0}}, nets, freq, device)

    half = len(val["sigma"]) // 2
    cal = {k: v[:half] for k, v in val.items()}
    tune = {k: v[half:] for k, v in val.items()}
    pc, pt = bundle.predict(cal["theta"], cal["mech"]), bundle.predict(tune["theta"], tune["mech"])
    floor = float(np.median(pt["spread"]))
    quantile = mt.conformal_quantile(np.abs(pc["sigma"] - cal["sigma"]) / (pc["spread"] + floor),
                                     coverage)  # fmt: skip
    conf = fb.confidence(pt["sigma"], pt["spread"], floor)
    level = factor * fb.threshold(conf, pt["sigma"], tune["sigma"])
    card = {
        "name": NAME, "version": version,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": "growth rate and frequency of the sub-synchronous mode of System B2 (task T1)",
        "architecture": {"arm": "R", "members": members, "width": (cfg or mt.TrainConfig()).width,
                         "depth": (cfg or mt.TrainConfig()).depth,
                         "steps": (cfg or mt.TrainConfig()).steps},
        "dataset": {"version": t1.VERSION, "spec_hash": t1.spec_hash(),
                    "train_cases": int(len(x) - h), "held_out_cases": int(h),
                    "validation_cases": int(len(val["sigma"]))},
        "training_commit": commit(),
        "inputs": input_ranges(),
        "rule": {"floor": floor, "level": level, "factor": factor,
                 "text": "outside the input ranges: solver. Otherwise confidence = |growth rate| "
                         "/ (spread + floor); at or below the level: solver."},
        "interval": {"coverage": coverage, "quantile": quantile,
                     "f_half_width_hz": mt.conformal_quantile(np.abs(pc["f"] - cal["f"]),
                                                              coverage)},
        "metrics": {},
    }  # fmt: skip
    bundle.card = card
    for name, d in (tests or {}).items():
        card["metrics"][name] = score(bundle, d)
    return bundle


def score(bundle: Bundle, d: dict) -> dict:
    p = bundle.predict(d["theta"], d["mech"])
    conf = bundle.confidence(p["sigma"], p["spread"])
    out = fb.outcome(fb.decide(conf, p["sigma"], bundle.card["rule"]["level"]), d["sigma"])
    err = np.abs(p["sigma"] - d["sigma"])
    return {
        "growth_rate_mae": float(err.mean()),
        "frequency_mae_hz": float(np.abs(p["f"] - d["f"]).mean()),
        "interval_coverage": float((err <= bundle.interval(p["sigma"], p["spread"])).mean()),
        **out,
    }  # fmt: skip


def save(bundle: Bundle, root: Path | None = None) -> Path:
    path = (root or ROOT) / bundle.card["name"] / bundle.card["version"]
    path.mkdir(parents=True, exist_ok=True)
    state = {"members": [m.state_dict() for m in bundle.members], "freq": bundle.freq.state_dict()}
    torch.save(state, path / "weights.pt")
    bundle.card["weights_sha256"] = hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest()
    (path / "card.json").write_text(json.dumps(bundle.card, indent=2))
    return path


def versions(name: str = NAME, root: Path | None = None) -> list[str]:
    base = (root or ROOT) / name
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "card.json").is_file())


def load(name: str = NAME, version: str | None = None, root: Path | None = None,
         device: str = "cpu") -> Bundle:  # fmt: skip
    """A registered model; the last version by name if none is given. Raises
    FileNotFoundError if there is none, and ValueError if the weights do not match the
    card or the card was made from another dataset specification."""
    found = versions(name, root)
    if not found:
        raise FileNotFoundError(f"no registered model {name!r} under {root or ROOT}")
    path = (root or ROOT) / name / (version or found[-1])
    card = json.loads((path / "card.json").read_text())
    blob = (path / "weights.pt").read_bytes()
    if hashlib.sha256(blob).hexdigest() != card["weights_sha256"]:
        raise ValueError("weights do not match the card")
    if card["dataset"]["spec_hash"] != t1.spec_hash():
        raise ValueError("the model was trained on another dataset specification")
    state = torch.load(path / "weights.pt", map_location=device, weights_only=True)
    arch = card["architecture"]

    def net(sd: dict) -> mt.MLP:
        m = mt.MLP(len(t1.SPEC), arch["width"], arch["depth"]).to(device)
        m.load_state_dict(sd)
        return m.eval()

    return Bundle(card, [net(sd) for sd in state["members"]], net(state["freq"]), device)
