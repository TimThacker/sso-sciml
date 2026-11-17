# 52: Summary of the screening approaches

Date: 2026-11-16 · Plan phase: 13

## Goal

One document naming the best deterministic and the best learned screen, with their operating points, what each needs and where each stops.

## What was done

- `docs/SCREENING.md`. No new computation: every number is quoted from the step that produces it (14, 18, 20, 21, 23 to 27, 29, 30, 40, 42, 47 to 51).

## Commands

None. The numbers come from the scripts named in each quoted step.

## Results

| | Screen | Operating point |
|---|---|---|
| Best deterministic | 2 × 2 admittance scan of the plant, least-squares fit, closed on the network at the 32 ends of its data's ranges; phasor-domain model beside it for the slow band | 0 unstable cases cleared in 24,659; clears 57.8 % of doubly-fed cases and 87.8 % / 70.6 % of converters, with a 1 % scan and the network to within 10 % (step 24) |
| Best learned | Hand calculation plus a learned correction, ensemble of five, fallback on the spread, range check | 0.72 % of cases to the solver, 0 of 1,376 unstable cases called stable, in the range (step 40); 0.9 % and 3.2 ms per answer in the service (step 42) |

- **The physics of the deterministic screen:** the resonance is a series resonance. At the frequency where the reactances cancel, the rotor resistance and the current controller's gain, divided by a negative slip, leave a negative resistance against the network's positive one. A test on the sum of impedances sees it; a test on the loop gain does not.
- **Phase 13 did not improve the learned screen.** Physics residuals as detectors (step 48) did no better than the spread, active learning (step 50) was cheaper but missed cases, and the converter (step 49) and multi-plant (step 51) versions did not beat their deterministic counterparts as screens.

## Disagreements

1. **The two screens are not scored on the same cases.** Step 24 used a random quarter of the cases of both plant types with uncertain network data; step 40 used the T1 test fold with the network exact. The table does not pair them.

## Decisions

- The deterministic scan screen is the reference screen. The learned screen is for throughput inside its range, with the solver behind it.
- Phase 13 is closed.

## Next

Open: validation rung 5 (recorded events) and a comparison with an electromagnetic-transient tool.
