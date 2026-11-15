"""Start the screening service.

    uv run scripts/serve.py                  # http://127.0.0.1:8000, docs at /docs
    uv run scripts/serve.py --port 8080

Uses the last model in the registry. If there is none, one is built first: from
data/t1-v2 if that dataset has been generated, otherwise from 16,384 cases generated on
the spot (about a minute of solver time, then training). scripts/phase11_build.py
builds the full model and writes its card.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import uvicorn

from sso_sciml import registry, service
from sso_sciml.data import t1

ROOT = Path(__file__).resolve().parents[1]


def model() -> registry.Bundle:
    try:
        return registry.load()
    except FileNotFoundError:
        print("no registered model; building one", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test, source = registry.dataset(ROOT / "data" / t1.VERSION)
    bundle = registry.build(train, val, "0.1.0-quick", device=device,
                            tests={"in_distribution": test})  # fmt: skip
    bundle.card["dataset"]["source"] = source
    registry.save(bundle)
    m = bundle.card["metrics"]["in_distribution"]
    print(f"built {bundle.card['name']} {bundle.card['version']} from {source}: error "
          f"{m['growth_rate_mae']:.3f} 1/s, solver share {100 * m['solver_share']:.1f} %",
          flush=True)  # fmt: skip
    return registry.load()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(service.create_app(model()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
