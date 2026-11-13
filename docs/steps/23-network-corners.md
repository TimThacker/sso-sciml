# 23: Network data with error: the ends of the ranges, not a wider margin

> **Restated 2026-11-13 (step 29).** The first pass behind this step took every fourth row of each set, which in a Sobol sequence is a corner of the set and not a sample of it. The pass was rerun on a random quarter (`src/sso_sciml/sampling.py`) and every number below is from the rerun. What changed is listed in `29-restated-with-delay.md`.

Date: 2026-11-12 · Plan phase: the standing screening question

## Goal

Step 22 found that network data error has to be covered by a margin of several 1/s, and that at 10 % error the margin needed clears almost nothing. It also found that the direction of the error is known for the doubly-fed plant. If each network quantity is known to within a bound, does running the screen at the ends of the ranges do better than a margin on the network as given?

## What was done

- `src/sso_sciml/neterror.py`: `corner_case`. The screen is given a network wrong by up to a bound in each of five quantities and is told the bound. It closes the exact plant model on the network as given, on all 32 ends of the five ranges, and on the one end that step 22 found unfavourable for the doubly-fed plant. The step 14 hand formula is evaluated as given and at that end.
- `scripts/network_corners.py`: the pass, the results file and two figures.
- `tests/test_neterror.py`: 1 more test.

## Commands

```powershell
uv run scripts/network_corners.py            # 24,659 cases, under half an hour on 8 processes
uv run scripts/network_corners.py --reuse    # numbers and figures from the saved pass
uv run pytest tests/test_neterror.py
```

## Setup of the study

- **Cases:** a quarter of the cases of both plant types, drawn at random, in and outside the training range together: 9,625 doubly-fed, 11,264 converters with a phase-locked loop, 3,770 without.
- **Error:** each of the five network quantities is wrong by a factor drawn evenly, on a logarithmic scale, between 1/(1 + b) and 1 + b, with b = 5 %, 10 % or 20 %. Unlike step 22 the error is bounded, and the screen knows b.
- **Screens:** the network as given with a margin from 0 to 10 1/s; the largest growth rate over the 32 ends, no margin; for the doubly-fed plant, the one end with less line reactance, less line resistance, more capacitor, a weaker parallel path and less plant-side impedance.
- **Band:** the sub-synchronous band for the doubly-fed plant, whose slow mode goes to the phasor-domain model (step 21). All modes for the converters (step 22).

## Results

All numbers are in `docs/results/network/network_corners.json`.

![Margin against the ends of the ranges](../results/network/fig60_network_corners.png)

Share of cases cleared, with the unstable cases cleared in brackets. "Margin as safe" is the smallest margin on the network as given that clears no more unstable cases than the 32 ends do.

| Plant, bound | Share stable | All 32 ends | Margin as safe | Network as given, that margin | Network as given, 2 1/s |
|---|---|---|---|---|---|
| Doubly-fed, 5 % | 73.4 % | 62.7 % (0) | 10 1/s | 9.5 % (0) | 60.9 % (1) |
| Doubly-fed, 10 % | | 58.0 % (0) | 10 1/s | 9.4 % (0) | 61.0 % (15) |
| Doubly-fed, 20 % | | 47.4 % (0) | 10 1/s | 9.4 % (0) | 61.9 % (77) |
| Converter with PLL, 5 % | 91.1 % | 89.6 % (0) | 5 1/s | 45.9 % (0) | 87.4 % (1) |
| Converter with PLL, 10 % | | 87.9 % (0) | 5 1/s | 45.7 % (0) | 87.4 % (5) |
| Converter with PLL, 20 % | | 84.8 % (0) | 10 1/s | 1.9 % (0) | 87.4 % (23) |
| Converter without PLL, 5 % | 77.6 % | 74.4 % (0) | 2 1/s | 71.0 % (0) | 71.0 % (0) |
| Converter without PLL, 10 % | | 70.4 % (0) | 10 1/s | 2.6 % (0) | 70.6 % (10) |
| Converter without PLL, 20 % | | 64.5 % (0) | 10 1/s | 2.5 % (0) | 70.7 % (26) |

- **The 32 ends clear no unstable case** on any plant at any bound: 24,659 cases, three bounds each.
- **They clear far more than a margin of the same safety.** With the network known to within 10 %, the ends clear 58 %, 88 % and 70 % of cases. The margin that is as safe (10, 5 and 10 1/s) clears 9 %, 46 % and 3 %.
- **20 % of error costs a converter 6 points.** The ends clear 84.8 % of converters with a phase-locked loop against 91.1 % that are stable.

### Why the ends do better

![The margin the ends imply](../results/network/fig61_implied_margin.png)

A margin is one number for every case. The ends give each case its own. How far the worst end lies above the network as given, at a bound of 10 %:

| | Median (1/s) | 95th percentile (1/s) |
|---|---|---|
| Doubly-fed plant | 2.1 | 6.4 |
| Converter with PLL | 0.03 | 4.2 |
| Converter without PLL | 0.18 | 8.5 |

- **Most converter cases are insensitive to the network and a few are very sensitive.** A margin sized for the few throws away the many. The ends find which is which.
- **For the doubly-fed plant the sensitivity is more even,** so the gain is smaller but still large: 58 % against 9 %.

### One end or all 32

For the doubly-fed plant the one unfavourable end clears 62.8 %, 58.1 % and 47.8 % at the three bounds, the same as all 32 to within 0.4 points.

- **It is not safe at any bound.** It clears 2, 1 and 8 unstable cases of 2,556 at 5 %, 10 % and 20 %.
- **It is the worst of the 32 in only 57 % to 66 % of cases.** The directions of step 22 are medians. In the rest another end is worse, usually by little.
- **All 32 cost 32 small eigenvalue problems,** so there is no reason to rely on the one.

### The hand formula at the unfavourable end

| Bound | Step 14 rule as given | Step 14 rule at the unfavourable end |
|---|---|---|
| 5 % | 56.2 % (1) | 50.9 % (1) |
| 10 % | 56.2 % (10) | 46.0 % (3) |
| 20 % | 56.6 % (51) | 37.2 % (4) |

- **Moving the hand formula to the unfavourable end helps and does not make it safe.** Its remaining misses are its own: the loops it leaves out, which step 14 covered with fitted margins. Those margins were set with exact network data.

### Is the worst end a bound?

Not strictly. The true growth rate exceeds the worst of the 32 ends in 2, 2 and 4 doubly-fed cases of 9,625 at the three bounds, and in no converter case. The growth rate is not monotone in every quantity, so the worst point of the box can lie inside it. None of those 8 cases led to an unstable case being cleared.

## What this says about screening

- **Uncertain network data call for the ends of the ranges, not a margin.** With the network known to within 10 % the screen clears 58 % to 88 % of cases and no unstable one, against 3 % to 46 % for a margin as safe.
- **How little information, again.** The screen needs each network number only as a range. Ten percent either way is enough to keep most of the cases an exact network would clear: 58 of 73 points for the doubly-fed plant, 87.9 of 91.1 and 70.4 of 77.6 for the converters.
- **This is the frequency-scan practice of sweeping outages, given a reason.** Running the study on a set of network conditions is a search over the ends of a range. What the result adds is that it replaces a margin; it does not go on top of one.
- **The complete screen for uncertain data.** For the resonance: plant admittance (a scan and a fit, or the matrices) closed on the network at the 32 ends of its ranges. For the slow band: the phasor-domain model, which hardly depends on the network.

## Disagreements

1. **Step 22 proposed the one unfavourable end.** It is the worst end in about 60 % of cases and clears unstable cases at every bound, 8 at 20 %. All 32 ends are used instead.
2. **The worst end is not a strict bound** (8 cases of 9,625, none of them an unstable case cleared). A search inside the box would close this. Not done.
3. **The error is bounded and even here, log-normal in step 22.** A screen at the ends of a range is only as good as the range. An error beyond the stated bound is not covered.
4. **The plant is exact and the scan is clean.** Scan noise and network error together were not run. The margin for scan noise (step 20) goes on top of the ends.
5. **Five quantities, 32 ends.** A real network has more uncertain elements and the count of ends doubles with each. Which few to vary is what step 22's sensitivity table is for: capacitor and resistance for a machine, reactance for a converter.
6. **The plant admittance is at the true operating point,** as in step 22.

## Decisions

- With uncertain network data the reference screen evaluates the ends of the ranges and takes the largest growth rate, with no margin for the network.
- The one unfavourable end is not used alone.
- The step 14 rule is kept for exact data and for data within 2 % (step 22). Beyond that the plant model at the ends replaces it.

## Next

In the order they bear on the result: scan noise and network ranges together; structured network error, one outage at a time; a source and a load on one bus; control delay and a filter capacitor in the converter. Still open from before: the protection model; a fit whose order grows with the data.
