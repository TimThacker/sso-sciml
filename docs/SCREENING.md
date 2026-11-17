# Screening for sub-synchronous oscillation: the best deterministic and the best learned screen

Date: 2026-11-16 · Plan phase: 13, step 52

A screen answers one question cheaply: can this case be cleared without a detailed study? It is judged by its tail first, the unstable cases it clears, and by its yield second, the share of cases it saves. This document names the best screen of each kind found in this project, gives its operating point, and says what it needs, why it works and where it stops. Every number is quoted from the step named beside it; each step names the script that reproduces it.

## In one table

| | Best deterministic screen | Best learned screen |
|---|---|---|
| What it is | The plant's 2 × 2 admittance scan, fitted by least squares and closed on the network at the 32 ends of the network data's ranges; the largest growth rate decides (step 24). For a doubly-fed plant, the phasor-domain model beside it for the slow band (step 21) | Hand calculation plus a learned correction (arm R), an ensemble of five perceptrons, with a fallback rule on the ensemble spread and a range check before it (steps 40, 42) |
| Needs | The plant as a black box at about 100 frequencies, good to 1 %; the network to within 10 %; no training | The plant model and about 100,000 solver runs inside a stated parameter range |
| Unstable cases cleared | **0 of 24,659 cases** (doubly-fed plant and two converter types) | **0 of 1,376** unstable test cases in the range |
| Yield | Clears 57.8 % of doubly-fed cases (73.4 % are stable), 87.8 % of converters with a PLL (91.1 %) and 70.6 % without (77.6 %) | Answers 99.3 % of cases itself and sends 0.72 % to the solver (step 40); the service sends 0.9 % (step 42) |
| Cost per case | 32 closed-loop evaluations of a fitted model, after one scan | 3.2 ms through the service, against 15 ms for the solver (step 42) |
| Outside its range | Has no training range. Holds with control delay (step 29), outages screened one by one (25), several plants at their joint operating point (27), shunt capacitance at both ends of its range (30) | Not safe without the range check: 16 of 1,476 unstable cases cleared at high compensation, 8 of 920 at low speed (step 40) |
| Use it for | The reference screen, any plant whose internals are not given, and anything outside a learned model's range | Throughput inside the range of a plant whose model is known, with the solver behind it |

The two do not compete for the same job. The learned screen saves more solver calls where it applies; the deterministic screen applies everywhere and needs far less about the plant.

## 1. The best deterministic screen

### What it is

1. Scan the plant at its terminals: the 2 × 2 admittance (D and Q axes) at 126 frequencies (step 24). Good to 1 %.
2. Fit a rational model to the scan by least squares (step 20). An exact interpolation through the points fails with any noise (step 19).
3. Close the fitted plant on the network, with each of the five uncertain network quantities of step 23 at both ends of its range: 2⁵ = 32 networks.
4. Take the largest growth rate of the closed loop over the 32. Below zero: clear. No margin is added when the scan is good to 1 % and the network to 10 % (step 24).
5. For a doubly-fed plant, also run the phasor-domain model of the planning study for the band below 2 Hz, with a margin of 0.05 1/s (step 21).

The Nyquist count on the same scans, with the network exact, gives the eigenvalue verdict in 33,579 of 33,580 cases; the exception grows at 0.001 1/s (step 18). Step 24 is that count made robust to a noisy scan and uncertain network data.

### Why it works: the physics

**Sub-synchronous resonance of a doubly-fed plant is a series resonance with a negative resistance in it.** The series capacitor and the line and machine inductances resonate at f_er below 60 Hz. At that frequency the rotor turns faster than the stator field, so the slip s = (f_er − f_r) / f_er is negative. The rotor's resistance, seen from the stator, is r / s. The rotor current controller acts as added rotor resistance (step 03), so r is the rotor resistance plus the controller's gain, and r / s is a large negative resistance. Where the reactances cancel, what is left is the network's resistance plus this negative one. If the sum is negative, the oscillation grows.

Four things follow, and the data confirms each:

- **The closer the resonance to the rotor frequency, the worse.** Small |s| makes r / s large. Unstable cases are 0.3 % of those resonating at 10 to 15 Hz and 33.8 % at 40 to 45 Hz (step 14).
- **The current-loop bandwidth is the one controller number that matters.** It sets r. Adding it to the network data and the rotor speed cuts the error of a growth-rate estimate from 2.47 to 0.67 1/s (step 14). The outer loops and the exact PLL tuning hardly matter; the PLL matters only as whether it is faster than the mode.
- **Network resistance matters as much as any reactance**, because it is the positive term in the sum (step 14).
- **A test on the size of the loop gain cannot see it.** At the frequency where the reactances cancel, the loop gain YZ is close to −1 whether the case is stable or not; what decides is the resistance left over. The passivity test clears 0.2 % of doubly-fed cases (step 18). The Nyquist count and the closed-loop growth rate are tests on the sum of the impedances, and they see it.

A grid-following converter on a weak grid is the other kind of instability: a loop whose gain is too high. There a small loop gain does rule it out, which is why the converters clear more than the machine at every rung (step 18).

### How little information

| Information | Screen | Cleared, doubly-fed plant | Unstable cleared | Step |
|---|---|---|---|---|
| Network data only (frequency scan of the network) | No resonance in the band | 24.8 % | 1 of 1,376 | 14 |
| Network R, X, C and three plant numbers (rotor speed, current-loop bandwidth, whether the PLL is faster than the mode) | Root of the loop impedance, margin −1 or −5 1/s by PLL speed | 79.9 % | 0 in the range; 5 of 8,673 outside | 14 |
| The same, plus the planning study's phasor-domain model for the slow band | Both together | 75.6 % | 0 of 1,854 (all modes) | 21 |
| A terminal scan with the network exact | Nyquist count | 85.9 % (all that is stable) | 0 in the range, 1 outside | 18 |
| A terminal scan good to 1 %, the network to within 10 % | Fitted scan at the 32 ends | 57.8 % | 0 of 24,659 | 24 |

The least information that decides the resonance is the loop's resistance at its resonance: the network's R, X and C, the rotor speed and the current-loop bandwidth. That is a hand calculation. It needs a margin because the plant's other loops move the root by up to 4.5 1/s when the PLL is faster than the mode (step 14). A scan replaces the three plant numbers when the plant is a black box; the ranges of the network data then cost more than the scan's noise (step 24: 58.0 % with a clean scan, 57.8 % with 1 % noise).

A positive-sequence stability model, the tool a planning study already has, reports every unstable resonance case as stable (steps 06, 14). It is exact for the slow control modes below 2 Hz, which a terminal scan barely sees (step 21). The two are blind in opposite bands; together they cover both.

### Where it stops, and what to add

| Situation | What goes wrong without care | What to do | Step |
|---|---|---|---|
| An outage | One case in ten stable at −10 1/s on the intact network is insecure after losing the parallel path | Screen each outage as its own network | 25 |
| A converter at another loading | One scan reused gives the wrong verdict in 7 % to 17 % of loadings | Two or three scans and interpolation at the operating point | 26, 31 |
| Several converters on a bus | Judging each alone misses 6 % to 25 % of unstable pairs | Scan each at the joint operating point and add the admittances | 27 |
| A network with shunt capacitance | A study without it misses 75 % to 85 % of unstable cases above 58 Hz | Include it, at both ends of its range | 30 |
| A scan with 5 % error | 7 unstable doubly-fed cases cleared at the ends | A margin of 1 1/s (5 1/s for a converter with a PLL) | 24 |
| A thin scan, points more than 2.5 % apart | Pointwise tests clear unstable cases | A fit, or a finer scan | 18, 19 |

## 2. The best learned screen

### What it is (step 40, served as `t1-screen 1.0.0` in step 42)

1. **Range check.** A parameter outside its training range: call the solver.
2. **Predict** with arm R, an ensemble of five perceptrons trained on 100,000 solver cases. Each predicts the correction to the hand calculation's growth rate (the root of the loop impedance above).
3. **Confidence** = |predicted growth rate| / (ensemble spread + 0.025 1/s). Above 2.77, the sign of the prediction is the answer. Otherwise call the solver. The threshold is the smallest that clears no unstable validation case, times 1.5.

### Operating point

On the in-distribution test fold of 13,105 cases, 1,376 unstable (step 40):

- 0.72 % of cases sent to the solver;
- 0 of 1,376 unstable cases called stable (95 % upper bound on the miss rate 0.22 %);
- 0 of 11,729 stable cases called unstable.

Through the service (step 42): 0.9 % to the solver, 0 of 1,376 missed, growth-rate error 0.069 1/s, 3.2 ms per answer.

### Why this form

- **The physics as the baseline beats the physics as inputs, and both beat none.** Arm R has the smallest error in distribution and on every outside set (0.071 1/s, against 0.090 for no physics; report section 3).
- **The physics as a loss is the worst use of it,** at five to eight times the cost (steps 34, 36, 37). Step 47 confirmed it on T2: the residual makes the data-trained network twice as wrong, and with no data it learns the steady state (error 0.99).
- **The spread is what makes it safe in the range.** With the margin alone the same model sends 3.0 % to the solver (step 40).

### What Phase 13 tried in order to improve it

| Idea | Result | Step |
|---|---|---|
| A physics residual at the predicted eigenvalue as the detector | No better than the spread. The impedance residual misses 19 unstable cases at high compensation where the spread misses 14; the state-matrix residual barely ranks the errors and costs most of a solve | 48 |
| Choosing the solver calls for training by the sign's uncertainty | 0.89 % to the solver with 4,000 labels, where random labels need 100,000 for 0.72 %; but 2 unstable cases missed in one of three repeats. Not adopted | 50 |
| The same recipe on the converter (System C), with the scan as inputs | 20 % to the solver in distribution, none missed; misses 1 unstable case on two of three outside sets. Clears no more than the deterministic scan screen | 49 |
| Several plants on a network, with a graph network | Trained on 2 to 4 buses, half the error of a plant-by-plant study up to 8 buses. Not yet a screen: the dataset is mostly unstable and the margin clears 3 % at most | 51 |

### Where it stops

- **Outside its range.** Without the range check it clears 16 of 1,476 unstable cases at high compensation and 8 of 920 at low speed. The two directions that fail without warning are series compensation above 80 % and rotor speed below 0.8 pu (step 40). No detector tried, learned or physical, closes this (steps 40, 48).
- **At the boundary.** Cases bisected to zero growth are sent to the solver 87.5 % to 99.9 % of the time, and the rule still lets 27 of 2,446 through (step 40).
- **A different plant.** It was trained on one plant model. A new plant needs a new dataset of solver runs.

For the response in time (task T2), the matching recommendation is the perceptron with the reduced model's response as inputs. Outside its range its ensemble spread flagged every response with more than 10 % error (step 48).

## 3. Recommendation

1. **Screen with the deterministic scan first** wherever the plant is a black box, a parameter is outside a learned model's range, or the network is being changed (outages, new plants, shunt capacitance). It needs a 1 % scan and the network to within 10 %, and it has cleared no unstable case in any study here.
2. **Use the learned screen for volume inside its range,** with the range check in front and the solver behind. It cuts the solver calls to under 1 % of cases.
3. **Keep the hand calculation in every output.** It is the learned screen's baseline, the only warning that does not come from training data (step 40), and the explanation an engineer can check: the loop resistance at the resonance.

## 4. What these numbers rest on

- Every number is against this project's solver on a representative plant assembled from typical values. The solver is checked against hand calculations, a benchmark's published frequencies, independent formulations and a second implementation of the full controls (steps 43, 44; `docs/VALIDATION.md`). Its eigenvalue for a published doubly-fed case was matched only after fitting one unpublished controller gain (step 43).
- No number has been compared with a recorded event (validation rung 5 is open).
- The yields depend on the parameter ranges sampled. The safety results, zero unstable cases cleared, are counts on those samples, not guarantees.
