# Model card: t1-screen 1.0.0

Written by `scripts/phase11_build.py` from the card in the registry. Do not edit by hand.

## What it is

- **Task:** growth rate and frequency of the sub-synchronous mode of System B2 (task T1).
- **Output:** growth rate (1/s) with an interval, frequency (Hz), and where the answer came from.
- **Form:** the hand calculation's growth rate plus a learned correction (arm R), mean of 5 perceptrons of 4 layers of 256; one more perceptron for the frequency.
- **Trained on:** dataset t1-v2 (specification hash `bb4ab735373a3947`), 80,000 cases, with 20,000 held out to pick the weights and 13,106 validation cases for the interval and the rule.
- **Training commit:** `52ca063923696eb9191dd3f79c2917a0a30a120e`.
- **Weights:** SHA-256 `49ff3e9f5b2d319b61c95825e6707fb6c7992e7e7bee8fe49fa76fe3e41779e1`. Not in the repository; rebuilt by the script above.

## Valid inputs

Outside any of these ranges the service answers from the solver, not from this model.

| Parameter | From | To | Meaning |
|---|---|---|---|
| `k` | 0.1 | 0.8 | series compensation as a fraction of line reactance |
| `w_r` | 0.8 | 1.2 | rotor speed, pu |
| `rsc_hz` | 2 | 30 | rotor current loop bandwidth, Hz |
| `pll_hz` | 5 | 30 | phase-locked loop natural frequency, Hz |
| `outer_ratio` | 0.03 | 0.15 | outer power loop crossover over current loop bandwidth |
| `ff` | 0 | 1 | terminal-voltage feed-forward gain of the grid-side converter |
| `x_line` | 0.5 | 1.5 | line reactance as a multiple of the reference 0.392 pu |
| `xr_line` | 10 | 30 | line X/R |
| `x_t` | 0.05 | 0.25 | plant-side series reactance, pu on the plant base |
| `par_ratio` | 0.5 | 8 | parallel path impedance as a multiple of the line's |
| `n_online` | 0.3 | 1 | fraction of units online |
| `r_r` | 0.004 | 0.008 | rotor resistance, pu |

## The fallback rule

Confidence = |growth rate| / (spread of the members + 0.0218 1/s). At or below 3.61 the solver is called. The level is 1.5 times the largest confidence at which a validation case had the wrong sign.

The interval of the growth rate is the prediction ± 1.92 × (spread + floor), built to hold 90 % of cases. The frequency is within 0.24 Hz in 90 % of validation cases.

## Measured

| Test set | Cases | Growth-rate error (1/s) | Frequency error (Hz) | Inside the interval | Sent to the solver | Unstable called stable | Stable called unstable |
|---|---|---|---|---|---|---|---|
| in distribution | 13,105 | 0.069 | 0.292 | 89.9 % | 0.9 % | 0 of 1,376 | 0 |
| boundary | 4,931 | 0.055 | 0.064 | 90.0 % | 93.8 % | 15 of 2,446 | 9 |
| high compensation | 4,096 | 0.233 | 0.177 | 63.2 % | 2.9 % | 14 of 1,476 | 0 |
| low speed | 4,096 | 0.248 | 0.220 | 78.6 % | 3.5 % | 5 of 920 | 0 |
| fast current loop | 4,095 | 0.180 | 0.112 | 73.7 % | 3.7 % | 0 of 2,276 | 0 |
| weak parallel path | 4,093 | 0.172 | 0.365 | 65.7 % | 2.4 % | 0 of 1,017 | 3 |
| other machine | 4,095 | 0.061 | 0.277 | 91.7 % | 0.9 % | 0 of 538 | 0 |

Only the first row is inside the valid inputs. The others show what the model would do if the range check were removed: it must not be.

## Limits

- In distribution the rule called 0 of 1,376 unstable test cases stable. That is a count, not a guarantee.
- Cases within the model's error of the stability boundary cannot be called by sign; the rule sends most of them to the solver and still misses some (row `boundary`).
- Above 80 % compensation and below 0.8 pu rotor speed the model is wrong without warning (step 40). The range check is what protects those cases.
- The plant is System B2 of this repository: one aggregated doubly-fed plant on a series-compensated line with a parallel path. It is not a model of any real plant.
- The hand calculation is returned with every answer as a second opinion that does not depend on the training data.

## Cost

- Answer from the model, through the service code: 3.2 ms a case, nearly all of it the hand calculation.
- Answer from the solver: 11.1 ms a case.
- Training: 52 s on cuda.
