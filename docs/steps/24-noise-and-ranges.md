# 24: Scan noise and uncertain network data together

> **Restated 2026-11-13 (step 29).** The first pass behind this step took every fourth row of each set, which in a Sobol sequence is a corner of the set and not a sample of it. The pass was rerun on a random quarter (`src/sso_sciml/sampling.py`) and every number below is from the rerun. What changed is listed in `29-restated-with-delay.md`.

Date: 2026-11-12 · Plan phase: the standing screening question

## Goal

Step 20 sized a margin to the noise on the scan. Step 23 replaced the margin for network data by the ends of the network ranges. A real study has both: a scan with error, and a network known to within a bound. Do the two remedies add, and what does the complete screen clear?

## What was done

- `src/sso_sciml/neterror.py`: `both_case`. A 126-point scan of the plant with noise is fitted by least squares (step 20). The fitted plant is closed on a network that is wrong by up to a bound in each of five quantities, as given and at the 32 ends of the ranges (step 23).
- `scripts/noise_and_ranges.py`: the pass, the results file and one figure.
- `tests/test_neterror.py`: 1 more test.

## Commands

```powershell
uv run scripts/noise_and_ranges.py            # 24,659 cases, about 25 minutes on 8 processes
uv run scripts/noise_and_ranges.py --reuse    # numbers and figure from the saved pass
```

## Setup of the study

- **Cases:** a random quarter of the cases of both plant types, as in steps 20 and 23: 9,625 doubly-fed (2,556 unstable in the sub-synchronous band), 11,264 converters with a phase-locked loop (997 unstable), 3,770 without (844 unstable).
- **Five combinations** of scan noise and network bound: clean and 10 %; 1 % and exact; 1 % and 5 %; 1 % and 10 %; 5 % and 10 %.
- **Screen:** the largest growth rate over the 32 ends, less a margin from 0 to 10 1/s. Sub-synchronous band for the doubly-fed plant, all modes for the converters.

## Results

All numbers are in `docs/results/network/noise_and_ranges.json`.

![Fitted noisy scan at the ends of the network ranges](../results/network/fig62_noise_and_ranges.png)

Share of cases cleared at the ends of the ranges, with the unstable cases cleared in brackets.

| Scan noise, network bound | Margin (1/s) | Doubly-fed plant | Converter with PLL | Converter without PLL |
|---|---|---|---|---|
| Share that is stable | | 73.4 % | 91.1 % | 77.6 % |
| Clean, 10 % | 0 | 58.0 % (0) | 88.0 % (0) | 71.0 % (0) |
| 1 %, exact | 0 | 73.1 % (163) | 91.1 % (8) | 77.7 % (3) |
| 1 %, exact | 0.5 | 65.9 % (0) | 86.8 % (1) | 73.8 % (0) |
| 1 %, 5 % | 0 | 62.6 % (1) | 89.5 % (0) | 74.4 % (0) |
| 1 %, 10 % | 0 | 57.8 % (0) | 87.8 % (0) | 70.6 % (0) |
| 5 %, 10 % | 0 | 57.4 % (7) | 87.9 % (6) | 70.9 % (2) |
| 5 %, 10 % | 1 | 53.7 % (0) | 81.3 % (2) | 63.8 % (0) |
| 5 %, 10 % | 5 | 31.9 % (0) | 38.6 % (0) | 25.9 % (0) |

- **With 1 % noise and the network within 10 % the ends of the ranges need no margin on top.** No unstable case is cleared on any plant. The spread over the ends is wider than the error the noise makes, so it covers both. With the network within 5 % the spread is narrower and 1 unstable doubly-fed case of 2,556 is cleared; a margin of 0.2 1/s removes it.
- **With an exact network the same scan does need a margin:** 163, 8 and 3 unstable cases are cleared without one. With 0.5 1/s none is on two plants, and 1 on the converter with a phase-locked loop. This is step 20 again.
- **With 5 % noise a margin of 1 1/s on top of the ends is enough on two plants of three.** Step 20 needed 5 1/s for the same noise with an exact network, because it had no other allowance. The converter with a phase-locked loop still needs 5 1/s: 2 unstable cases are cleared at 1 1/s and 1 at 2 1/s.
- **Noise costs nothing in cleared cases once the ranges are in.** At a bound of 10 % the clean scan clears 58.0 %, 88.0 % and 71.0 %; the scan with 1 % noise clears 57.8 %, 87.8 % and 70.6 %.

### Without the ends

On the network as given, the smallest margin that clears no unstable case is 2 to 10 1/s over the plants and the combinations with a network bound, and at that margin 2 % to 61 % of cases are cleared. A margin has to cover the worst case in the whole set; the ends cover each case by its own sensitivity.

## What this says about screening

- **The complete screen, as it now stands.** A 2 × 2 scan of the plant at about a hundred frequencies, good to 1 %. A least-squares fit. The fitted plant closed on the network at the 32 ends of the ranges of its data. The largest growth rate decides, with no margin. For a plant with slow control modes, the phasor-domain model beside it.
- **What it clears with a 1 % scan and a network known to within 10 %:** 58 % of doubly-fed cases against 73 % that are stable in the band, 87.8 % of converters with a phase-locked loop against 91.1 %, and 70.6 % of those without against 77.6 %. No unstable case in 24,659.
- **How little information, in one line.** The plant to 1 % at a hundred frequencies, the network to 10 %. The second is the binding one for the machine. A converter loses 3 to 7 points to the two together.
- **Allowances do not stack.** A margin for noise and a spread for network data are both allowances for error in the growth rate. The larger one covers the smaller unless the smaller is of the same size, as at 5 % noise.

## Disagreements

1. **Step 23 said the margin for scan noise goes on top of the ends.** At 1 % noise and a bound of 10 % it is not needed. At 5 % it is: 1 1/s for the doubly-fed plant and the converter without a phase-locked loop, and the 5 1/s of step 20 for the converter with one.
2. **With 5 % noise and no margin 7 unstable doubly-fed cases and 8 converter cases are cleared at the ends.** The fit is off by more than the spread of the ends in those cases.
3. **The ends are safe here because the network error is inside its bound.** As in step 23.
4. **A quarter of the cases, one noise draw and one network draw per case and combination.** Step 29 repeated the combination of 1 % and 10 % on the converters with another draw and cleared no unstable case.

## Decisions

- The reference screen for a scan with error and a network with ranges is the least-squares fit closed on the 32 ends, with no margin for scan error up to 1 % where the network bound is 10 %. At 5 % scan error the margin is 1 1/s for the doubly-fed plant and the converter without a phase-locked loop and 5 1/s for the converter with one.
- Margins from steps 20 and 22 apply only where one of the two error sources is absent.

## Next

Structured network error, one outage at a time, which is step 25. Then a source and a load on one bus; control delay and a filter capacitor in the converter. Still open from before: the protection model; a fit whose order grows with the data.
