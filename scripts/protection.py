"""What the doubly-fed plant does after the first limit, with protection modelled.

    uv run scripts/protection.py            # four runs of 5 s, 384 cases each, on the GPU
    uv run scripts/protection.py --reuse    # numbers and figures from the saved runs

Needs the dataset from scripts/phase3_generate_t1.py. 192 unstable and 192 stable cases
of the T1 test fold are run for 5 s after an offset of the series capacitor voltage of
0.01, 0.1 and 0.3 pu, with a rotor voltage limit, a rotor current limit, anti-windup and
a DC chopper (`sso_sciml.protection`). One more run at 0.1 pu has the rotor voltage limit
only, which is the model of step 16. Each run is read for the conditions that take a
plant off the network, and for what the oscillation does if none is met. The time of
the first trip is also found from the modes, with no time stepping.

Writes data/protection-v1/<run>.npz (not committed), docs/results/phase3/protection.json
and figures fig79 and fig80. Every number in docs/steps/33-protection.md comes from this
script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sso_sciml import protection as prot
from sso_sciml import report
from sso_sciml.data import t1 as data
from sso_sciml.solver.linearize import jacobian
from sso_sciml.solver.modal import linear_response

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / data.VERSION
DATA = ROOT / "data" / "protection-v1"
OUT = ROOT / "docs" / "results" / "phase3"
N_EACH = 192
T_END, DT, EVERY = 5.0, 5e-4, 2
VC = 4
RUNS = {
    "protected, 0.01 pu": (prot.SETTINGS, 0.01),
    "protected, 0.1 pu": (prot.SETTINGS, 0.1),
    "protected, 0.3 pu": (prot.SETTINGS, 0.3),
    "limit only, 0.1 pu": (prot.LIMIT_ONLY, 0.1),
}
COL = {name: j for j, name in enumerate(prot.SIGNALS)}
CAUSES = (*prot.TRIPS, "run failed")
FATES = ("sustained", "growing", "decaying", "decayed")


def pick() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(SOURCE / "in_distribution.npz")
    ok = (z["solver"][:, 4] > 0) & (z["fold"] == 2)
    top = z["solver"][:, 2]
    rows = np.concatenate([np.where(ok & (top > 0))[0][:N_EACH],
                           np.where(ok & (top < 0))[0][:N_EACH]])  # fmt: skip
    return z["theta"][rows], z["solver"][rows]


def file_of(name: str) -> Path:
    return DATA / (name.replace(", ", "_").replace(" ", "_").replace(".", "p") + ".npz")


def simulate(theta: np.ndarray, device: str) -> dict:
    ops = [data.operating_point(row) for row in theta]
    model = ops[0][0]
    x_eq = torch.stack([o[1] for o in ops])
    base = {k: torch.stack([o[2][k] for o in ops]).to(device) for k in ops[0][2]}
    steady = prot.signals(model, x_eq.to(device), base).cpu().numpy()
    # The same from the modes: one response per case, scaled by the offset.
    a = np.array([jacobian(m.rhs, x, p).numpy() for m, x, p in ops])
    t = np.arange(int(round(T_END / DT)) // EVERY + 1) * (DT * EVERY)
    dx0 = np.zeros(data.N_STATE)
    dx0[VC] = 1.0
    unit = np.array([linear_response(a[i], dx0, t) for i in range(len(ops))]).transpose(1, 0, 2)
    cost = {}
    for name, (settings, eps) in RUNS.items():
        p = prot.with_settings(base, settings)
        x0 = x_eq.clone()
        x0[:, VC] += eps
        t0 = time.perf_counter()
        t, sig, failed = prot.run(model, x0.to(device), p, T_END, DT, EVERY)
        cost[name] = time.perf_counter() - t0
        lin = []
        with np.errstate(all="ignore"), torch.no_grad():
            for chunk in np.array_split(np.arange(len(t)), 25):
                x = torch.as_tensor(eps * unit[chunk] + x_eq.numpy()[None]).to(device)
                lin.append(prot.signals(model, x, base).cpu().numpy())
        np.savez_compressed(file_of(name), t=t, sig=sig.astype(np.float32), failed=failed,
                            lin=np.concatenate(lin).astype(np.float32), steady=steady)  # fmt: skip
        print("done", name, f"{cost[name]:.0f} s", flush=True)
    return cost


def quantiles(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(q) for q in np.quantile(x, [0.05, 0.5, 0.95])] if x.size else []


def modal_error(first: np.ndarray, modal: np.ndarray, g: np.ndarray) -> dict:
    both = g & np.isfinite(first) & np.isfinite(modal)
    err = (modal - first)[both]
    out = {
        "cases_the_modes_also_trip": int(both.sum()),
        "modes_trip_and_run_does_not": int((g & ~np.isfinite(first) & np.isfinite(modal)).sum()),
        "run_trips_and_modes_do_not": int((g & np.isfinite(first) & ~np.isfinite(modal)).sum()),
    }
    if err.size:
        out.update({
            "error_median_s": float(np.median(err)),
            "error_median_abs_s": float(np.median(np.abs(err))),
            "error_p95_abs_s": float(np.quantile(np.abs(err), 0.95)),
            "error_relative_median": float(np.median(np.abs(err) / first[both])),
            "late_share": float((err > 0).mean()),
            "late_by_more_than_0.1_s": int((err > 0.1).sum()),
        })  # fmt: skip
    return out


def study(name: str, groups: dict[str, np.ndarray]) -> tuple[dict, dict]:
    z = np.load(file_of(name))
    t, sig, lin = z["t"], z["sig"].astype(float), z["lin"].astype(float)
    eps = RUNS[name][1]
    trips = prot.trip_times(t, sig, z["failed"])
    modal = prot.trip_times(t, lin)
    first = trips["first"]
    tripped = np.isfinite(first)
    cause = np.array([min(CAUSES, key=lambda c, j=j: np.nan_to_num(trips[c][j], nan=np.inf))
                      if tripped[j] else "" for j in range(len(first))])  # fmt: skip
    # What the oscillation is doing at the end, for the cases still connected.
    line = sig[:, :, COL["i_line"]]
    early = prot.swing(t, line, 0.0, 0.2)
    mid = prot.swing(t, line, T_END - 1.0, T_END - 0.5)
    end = prot.swing(t, line, T_END - 0.5, T_END)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = end / mid
        fate = np.select([end < 0.1 * early, ratio > 1.1, ratio < 0.9],
                         ["decayed", "growing", "decaying"], "sustained")  # fmt: skip
    fate = np.where(tripped, "", fate)
    limits = {
        "rotor voltage limit": prot.first_outside(t, sig[:, :, COL["v_command"]], 0.0,
                                                  RUNS[name][0]["vr_max"], 0.0),
        "rotor current limit": prot.first_outside(t, sig[:, :, COL["i_reference"]], 0.0,
                                                  RUNS[name][0].get("ir_max", 1e6), 0.0),
        "chopper": prot.first_outside(t, sig[:, :, COL["p_chopper"]], -1.0, 1e-9, 0.0),
    }  # fmt: skip
    first_limit = np.fmin.reduce(list(limits.values()))
    res: dict = {"offset_pu": eps}
    for group, g in groups.items():
        res[group] = {
            "cases": int(g.sum()),
            "tripped": int((g & tripped).sum()),
            "first_trip": {c: int((g & (cause == c)).sum()) for c in CAUSES},
            "trip_time_s": quantiles(first[g]),
            "still_connected": {f: int((g & (fate == f)).sum()) for f in FATES},
            "line_current_swing_at_the_end_pu": {
                f: quantiles(end[g & (fate == f)]) for f in ("sustained", "growing")},
            "a_limit_acts_before_the_trip": int((g & tripped & (first_limit <= first)).sum()),
            "a_limit_acts_and_no_trip": int((g & ~tripped & np.isfinite(first_limit)).sum()),
            "limit_before_trip_by_s": quantiles((first - first_limit)[g & tripped
                                                                      & (first_limit <= first)]),
            "trip_from_the_modes": modal_error(first, modal["first"], g),
        }  # fmt: skip
    # Other levels for the rotor current trip, and none at all, on the first group.
    g = next(iter(groups.values()))
    res["rotor current trip level"] = {}
    for level in (1.5, 2.0, 3.0, None):
        other = dict(prot.TRIPS)
        if level is None:
            del other["rotor current"]
        else:
            other["rotor current"] = ("i_rotor", 0.0, level, 0.0)
        tr, mo = prot.trip_times(t, sig, z["failed"], other), prot.trip_times(t, lin, None, other)
        hit = np.isfinite(tr["first"])
        names = (*other, "run failed")
        why = np.array([min(names, key=lambda c, j=j: np.nan_to_num(tr[c][j], nan=np.inf))
                        if hit[j] else "" for j in range(len(hit))])  # fmt: skip
        res["rotor current trip level"]["none" if level is None else f"{level:g} pu"] = {
            "tripped": int((g & hit).sum()),
            "first_trip": {c: int((g & (why == c)).sum()) for c in names},
            "trip_time_s": quantiles(tr["first"][g]),
            "trip_from_the_modes": modal_error(tr["first"], mo["first"], g),
        }
    res["steady_state"] = {
        "rotor current reference, max": float(z["steady"][:, COL["i_reference"]].max()),
        "rotor voltage command, max": float(z["steady"][:, COL["v_command"]].max()),
    }
    keep = {"t": t, "sig": sig, "first": first, "modal": modal["first"], "cause": cause,
            "fate": fate, "tripped": tripped}  # fmt: skip
    return res, keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    theta, solver = pick()
    top = solver[:, 2]
    sso = solver[:, 0] > 0
    groups = {"sub-synchronous mode unstable": sso,
              "slow mode unstable only": (top > 0) & ~sso,
              "stable": top < 0}  # fmt: skip
    res: dict = {"cases": {k: int(g.sum()) for k, g in groups.items()},
                 "window_s": T_END, "settings": prot.SETTINGS,
                 "trips": {k: list(v) for k, v in prot.TRIPS.items()},
                 "growth_rate_1_per_s": {k: quantiles(top[g]) for k, g in groups.items()},
                 "runs": {}}  # fmt: skip
    if not args.reuse:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        res["cost_s"] = simulate(theta, device)
    elif (OUT / "protection.json").exists():
        res["cost_s"] = json.loads((OUT / "protection.json").read_text()).get("cost_s")
    kept = {}
    for name in RUNS:
        res["runs"][name], kept[name] = study(name, groups)
        print(name, json.dumps(res["runs"][name], indent=1))
    (OUT / "protection.json").write_text(json.dumps(res, indent=2))

    # ---------------- figure: what becomes of each case
    parts = [*CAUSES, *FATES]
    colours = [*report.SERIES[:5], "#c9c7bd", "#898781", "#e1e0d9", "#f1f0ea"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8), facecolor=report.SURFACE)
    for ax, group in zip(axes, groups, strict=True):
        bottom = np.zeros(len(RUNS))
        for part, colour in zip(parts, colours, strict=True):
            vals = np.array([
                (r[group]["first_trip"].get(part, 0) if part in CAUSES
                 else r[group]["still_connected"][part]) for r in res["runs"].values()
            ], dtype=float)  # fmt: skip
            label = f"trip: {part}" if part in prot.TRIPS else part
            ax.bar(range(len(RUNS)), vals, bottom=bottom, color=colour, label=label, width=0.6)
            bottom += vals
        ax.set_xticks(range(len(RUNS)), [n.replace(", ", "\n") for n in RUNS], fontsize=9)
        ax.set_title(f"{group[0].upper() + group[1:]}", loc="left", fontsize=10)
        report.style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"Cases, by what happens within {T_END:g} s")
    axes[2].legend(frameon=False, fontsize=8, labelcolor=report.LABEL, loc="upper left",
                   bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    report.title(fig, "After a capacitor-voltage offset: first trip, or the state of the "
                 "oscillation at the end")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig79_protection_outcomes.png", dpi=160)
    plt.close(fig)

    # ---------------- figure: time of the first trip from the modes, and one case
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), facecolor=report.SURFACE)
    for name, colour in zip(list(RUNS)[:3], report.SERIES, strict=False):
        k = kept[name]
        both = sso & k["tripped"] & np.isfinite(k["modal"])
        axes[0].scatter(k["first"][both], k["modal"][both], s=14, color=colour, lw=0, label=name)
    axes[0].plot([0, T_END], [0, T_END], color=report.INK, lw=0.8, ls="--")
    axes[0].set_xlabel("First trip in the run (s)")
    axes[0].set_ylabel("First trip from the modes (s)")
    axes[0].set_title("Cases with an unstable sub-synchronous mode", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=report.LABEL)
    k = kept["protected, 0.1 pu"]
    cand = np.where(sso & k["tripped"])[0]
    i = int(cand[np.argsort(k["first"][cand])[len(cand) // 2]]) if cand.size else 0
    stop = len(k["t"])
    if cand.size:
        stop = int(np.searchsorted(k["t"], min(T_END, k["first"][i] + 0.1)))
    for signal, colour in zip(("v_dc", "i_rotor", "v_terminal", "p_chopper"), report.SERIES,
                              strict=False):  # fmt: skip
        axes[1].plot(k["t"][:stop], k["sig"][:stop, i, COL[signal]], color=colour, lw=1.1,
                     label=signal.replace("_", " "))  # fmt: skip
    if cand.size:
        axes[1].axvline(k["first"][i], color=report.INK, lw=0.8, ls=":")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("pu")
    title = f"One case, growth rate {top[i]:+.1f} 1/s, offset 0.1 pu"
    if cand.size:
        title += f". Dotted: trip on {k['cause'][i]}"
    axes[1].set_title(title, loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=report.LABEL)
    for ax in axes:
        report.style(ax)
    report.title(fig, "When the plant trips")
    fig.tight_layout()
    fig.savefig(OUT / "fig80_trip_time.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
