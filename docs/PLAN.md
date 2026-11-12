# Project plan

Status: Phases 0 and 1 complete (2026-11-11). Target numbers for later phases are targets, not results.

Revision 2 (2026-11-11). Changes from revision 1, and why:

- **The object of study is the physical mechanism, not the impedance scan.** An impedance scan is a way of analysing a linearised model. The physics is that a rotor turning faster than a sub-synchronous stator field sees negative slip, reflects negative resistance into the network, and feeds energy into the electrical resonance. The learning targets are now the quantities of that mechanism (slip at resonance, negative resistance, net resistance, growth rate, energy exchanged) and the time-domain response. The impedance scan is kept only as one cross-check of the solver in Phase 2.
- **Physics-informed neural networks get their own phase**, in forward, parametric and inverse forms.
- **A controlled comparison of where the physics goes** is now the centre of the ML work: in the loss, in the input features, in the architecture, or nowhere.
- Phase 1 was executed. Its main finding changed the emphasis of Phase 2 (see Phase 1 below).

## 1. What this project is

Sub-synchronous oscillation (SSO) on series-compensated lines, studied through its mechanism, with learned models built next to a physics solver.

The question the project keeps asking is the one an engineer asks: **will this machine, at this operating point, on this network, see negative slip at the electrical resonance and feed an oscillation, and how fast will it grow?**

The project builds:

1. A reference solver written from first principles, with network electromagnetic dynamics kept, validated against published benchmarks, hand calculations, an independent formulation, and recorded events.
2. The hand calculations themselves (slip, negative resistance, damping torque) as code, with their range of validity measured against the solver.
3. Several kinds of learned model trained on the solver, differing in where the physics enters.
4. A measured account of how each fails outside its training range, how the failure is detected, and when the system falls back to the solver.
5. A design search over mitigation settings.
6. A versioned service with tests that fail when a number is wrong.

### Why SSO rather than GIC

- **The solver is worth replacing.** A GIC study is one sparse DC solve per storm scenario. An SSO study needs time-domain electromagnetic simulation across operating points, contingencies and controller settings.
- **It covers more of the target scope.** Machine electromagnetics, converter dynamics, and interconnection practice meet in SSO.
- **There is published ground truth** to check the solver against before any learning.

### Scope boundaries

In scope: balanced three-phase dynamics in the dq frame, 0.1 to 120 Hz, small-signal and large-signal. Out of scope for now: unbalanced faults, switching-level converter models, saturation, protection logic. Each is an entry in `docs/ASSUMPTIONS.md`.

## 2. How the plan maps to the scope document

| Scope item | Where it is addressed |
|---|---|
| Physics models you set up yourself; know which assumptions and discretization choices drive the answer | Phases 1 and 2. `docs/ASSUMPTIONS.md`, `docs/EQUATIONS.md` |
| Checked against experiments or measured data; say what you did when they disagreed | Phase 2 validation ladder; "Disagreements" section in every step log |
| Surrogates, neural operators, physics-informed networks | Phases 4 to 7 |
| Out-of-distribution behaviour, detection, solver fallback | Phase 9, on extrapolation splits fixed in Phase 3 |
| Decide where solver fidelity is required and where a learned model is enough | Phase 9 fallback policy; Section 9 |
| Search a design space | Phase 10 |
| Something they can run; behind an API, versioned, tests that catch a wrong number | Phase 11 |
| Training runs designed and debugged in PyTorch; uncertainty quantification | Phases 4 to 8 |
| Standards the domain works to | Section 10 |

## 3. The physics

### The mechanism

A series-compensated line resonates electrically at `f_er = f0 √(X_C / X)`, below the system frequency. A current at `f_er` sets up a stator field turning slower than the rotor. The rotor circuits see slip

```
s = (f_er − f_r) / f_er   < 0
```

Rotor resistance reflected to the stator is proportional to `R_r / s`, so it is negative. Three things can then happen.

1. **Induction generator effect.** If the negative resistance exceeds the network resistance at `f_er`, the electrical resonance grows by itself. No shaft motion is needed.
2. **Torsional interaction.** If `f0 − f_er` lands on a shaft torsional frequency, rotor oscillation and sub-synchronous current reinforce each other through the air-gap torque. The electrical side supplies negative damping to the shaft.
3. **Control interaction.** In a doubly-fed machine the rotor-side converter regulates rotor current. Its proportional gain acts like added rotor resistance, so the negative resistance at sub-synchronous slip is multiplied. This is the mechanism of the wind-plant events.

Each has a hand calculation (`physics/slip.py`, `physics/torsional.py`) and each is a limit of the full dynamic model. The hand calculations serve three purposes: validation of the solver, physical features for the learned models, and the plain-language explanation attached to every prediction.

### Mechanism quantities

These are computed for every case and are both inputs to and targets for the learned models.

| Quantity | Meaning |
|---|---|
| `f_er` | electrical resonance frequency |
| slip at resonance | how far below rotor speed the resonance sits |
| `R_machine` | resistance the machine presents at `f_er` (negative) |
| `R_net` | loop resistance at `f_er`; self-excitation when negative |
| `σ_hand` | growth rate from `R_net` and the reactance slope |
| `D_e(f_n)` | electrical damping torque at each torsional frequency |
| sub-synchronous energy per cycle | energy the machine delivers to the network at `f_er`, from the time-domain run |
| `σ`, `f` | growth rate and frequency from the full model |

### Systems

**System A: IEEE First Benchmark Model (done, Phase 1).** Synchronous generator, six-mass shaft, series-compensated line. Negative slip is always present but the negative resistance is five times too small to self-excite; the instability is torsional interaction.

**System B1: fixed-speed induction generator on a series-compensated line (Phase 2, done).** The cleanest case of mechanism 1. The slip calculation is exact for a round rotor, and the hand calculation and the solver agree on the boundary. The expectation that self-excitation would appear at realistic compensation did not hold for the base machine (boundary at 96.5 %); it needs more rotor resistance or a slower rotor.

**System B2: doubly-fed induction generator with converter controls (Phase 2, main study).** Mechanism 3. About 24 states. Parameter space of about 14 dimensions:

| Group | Parameters |
|---|---|
| Network | compensation level, line R/X, line length, grid short-circuit ratio |
| Operating point | wind speed (sets rotor speed, hence slip), turbines online, terminal voltage, reactive dispatch |
| Controls | rotor-side current loop Kp and Ki, outer loop bandwidth, PLL bandwidth, grid-side current loop Kp |
| Machine | rotor resistance, leakage reactances |

Trends the solver must reproduce before data generation: damping falls as compensation rises, as wind speed drops (slip at resonance moves toward zero from below, so `R_r / s` grows), and as rotor-side current loop gain rises.

**System C: a generic converter on a weak or series-compensated network (done, step 17).** No rotor and no slip. Source or load, with a phase-locked loop or without. The oscillation comes from the loop that holds power constant meeting the network impedance. A deliberately different mechanism, to test whether the screening result transfers.

### Solver

One model definition in PyTorch, float64, gives:

1. Nonlinear time-domain simulation (batched fixed-step RK4; SciPy Radau and DOP853 as references).
2. Small-signal model: analytical steady state, Jacobian by automatic differentiation, eigenvalues and participation factors on the CPU.
3. A right-hand side that is differentiable with respect to state and parameters. This is what the physics-informed losses use.

## 4. Hardware and compute budget

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3070, 8 GB (about 6.9 GiB free with the desktop running) |
| CPU | Intel i7-10700K, 8 cores / 16 threads |
| RAM | 32 GB |
| Disk budget for data | 20 GB |
| Stack | Python 3.12, PyTorch 2.14.1 + CUDA 13.0, managed by `uv` |

Measured so far:

- Mixed-precision MLP training: about 3.1 million samples per second on the GPU, 27 times the CPU (`docs/steps/00-setup.md`).
- Batched eigenvalue solves: 20 to 60 times slower on the GPU. They run on the CPU.
- Batched RK4 on the 20-state model: the GPU is slower below about 5,000 trajectories per batch and 4 times faster at 32,768 (`docs/steps/01-solver-core-fbm.md`).

The GPU does all training and inference, and time-domain data generation in large batches. Models are sized for 8 GB with mixed precision. Physics-informed training needs float64 residuals in places; those parts run without mixed precision and are budgeted separately. Estimated GPU time for the plan is under 120 hours; no single run is planned beyond 8 hours.

## 5. Learned models

### Tasks

| Task | Input → output |
|---|---|
| **T1 Mechanism and margin** | parameters θ → slip at resonance, `R_net`, growth rate and frequency of the critical modes, stable/unstable |
| **T2 Time-domain response** | θ and a disturbance → state trajectories, including growth and the limit cycle once limits act |
| **T3 Measurement inverse** | noisy waveform window → mode frequency, growth rate, and plant parameters |
| **T4 Design** | search over mitigation settings for worst-case damping |

### Model families

| Family | Used for | Members |
|---|---|---|
| Classical baselines | T1, T3 | gradient-boosted trees, Gaussian process, Prony / matrix pencil |
| Plain networks | T1, T2 | MLP, deep ensemble |
| Neural operators | T2 | DeepONet (branch θ, trunk t), Fourier neural operator in time |
| Sequence models | T2, T3 | temporal convolutional network, small transformer, 1-D CNN |
| Physics-informed networks | T1, T2, T3 | PINN (see below) |
| Hybrid / structured | T2 | neural ODE with the known right-hand side plus a learned correction |

### Where the physics goes: the central comparison

Five ways of using the same physical knowledge, compared with architecture size, data, and splits held equal.

| Arm | Physics enters as | Concretely |
|---|---|---|
| **N** None | nothing | raw parameters θ in, target out |
| **F** Feature | extra inputs | θ plus the mechanism quantities from the hand calculation: `f_er`, slip at resonance, `R_machine`, `R_net`, `σ_hand`, `D_e(f_n)` |
| **R** Residual target | the output baseline | the network learns the correction to the hand estimate: `σ = σ_hand + net(θ)` |
| **L** Loss | a penalty during training | data loss plus a physics residual. For T1: the eigen-residual `‖A(θ) v̂ − λ̂ v̂‖`, which needs no labels. For T2: the ODE residual `‖dx̂/dt − f(x̂, θ)‖` at collocation points |
| **A** Architecture | a hard constraint | the known `f(x, θ)` inside a neural ODE with a learned correction term; initial conditions imposed exactly |

Plus the combination **F+L**.

Measured for every arm:

- Error in distribution, at data budgets of 100, 1,000, 10,000 and 100,000 labelled cases. The interesting region is the small-data end.
- Error on each extrapolation split.
- Whether the prediction still obeys the mechanism where it should: sign of growth rate against sign of `R_net` for System B1; monotonic trends in compensation and wind speed.
- Training cost in GPU-hours and sensitivity to the loss weight (arm L) or to an error in the feature (arm F).
- **Feature corruption test for arm F.** The hand calculation is known to be wrong in places: it overestimates the mode 1 growth peak of System A by 86 % (Phase 1). Does a model fed that feature inherit the error or learn to discount it? Arms F and R are evaluated inside and outside the hand calculation's range of validity.
- **Label-free test for arm L.** How far does the physics loss get with no solver labels at all, and how many labels does it take for arm N to catch up?

Hypotheses, written before any training:

1. With little data, F and R beat N by a wide margin, because the hand calculation already carries most of the answer for mechanism 1.
2. F and R degrade where the hand calculation is invalid (strong coupling, converter limits), and R degrades more than F because it cannot ignore the baseline.
3. L extrapolates better than N in parameter directions the residual covers, at several times the training cost.
4. L is harder to train on the time-domain task than on the eigen task, because of stiffness and oscillation (below).
5. With abundant data the arms converge in distribution and differ only out of distribution.

Each hypothesis gets a verdict in the Phase 6 report.

### PINNs in detail

Three forms, each a standing topic in Phase 5.

1. **Forward PINN, one case.** A network `x̂(t)` for one parameter set, trained on the ODE residual and initial condition only. It replaces nothing useful, since RK4 solves the same case in a second. Its job is to find out what it takes to train a PINN on this system at all.
2. **Parametric PINN.** `x̂(t; θ)` over a range of parameters. This is the form that could act as a surrogate. Compared directly with DeepONet and FNO trained on solver data, and with a PINN given both.
3. **Inverse PINN.** Unknown plant parameters become trainable variables; the network fits a measured waveform while satisfying the ODE. Compared in Phase 8 with gradient-based identification straight through the RK4 solver, which uses the same physics with no network.

Known difficulties on this system, and the plan for each:

| Difficulty | Why it applies here | Plan |
|---|---|---|
| Spectral bias | The solution oscillates at 10 to 100 Hz for seconds; a plain MLP learns low frequencies first | Fourier feature inputs; sinusoidal activations; compare |
| Stiffness and scale | States range from fluxes near 1 to speed deviations near 1e-4; rates range from 0.3 to 600 1/s | Non-dimensionalise each state and residual; per-equation residual weights |
| Causality | Residual minimised over the whole window lets late times converge to a wrong trajectory | Causal weighting; time-marching over windows |
| Loss balancing | Data, residual, and initial-condition terms differ by orders of magnitude | Gradient-norm balancing against fixed weights; report sensitivity |
| Unstable cases | A growing oscillation amplifies any early error | Train on log-envelope and phase as an alternative representation |
| Trivial solution | The equilibrium satisfies the ODE exactly | Hard-constrain the initial condition: `x̂(t) = x0 + t · net(t)` |
| Memory | Residuals need derivatives of the network output with respect to time | Forward-mode derivatives; collocation batch sized for 8 GB |

A negative result is an acceptable outcome of Phase 5. If a parametric PINN cannot match a data-trained operator at equal GPU time, the report says so with the numbers.

### Uncertainty

Deep ensembles, MC dropout, and split conformal prediction on T1 and T2; the Gaussian process brings its own posterior. Calibration reported in and out of distribution.

## 6. Phases

Every phase ends with a step log in `docs/steps/`, passing tests, and a commit. Effort is in part-time weeks for one person.

### Phase 0: Setup (done)

See `docs/steps/00-setup.md`.

### Phase 1: Solver core and System A (done)

See `docs/steps/01-solver-core-fbm.md`. Summary:

- [x] Equations and conventions written down (`docs/EQUATIONS.md`).
- [x] Model interface, steady state, autodiff linearisation, eigenvalues, batched RK4, reference integrators, matrix pencil.
- [x] Five torsional frequencies within 0.005 Hz of published values.
- [x] Slip calculation implemented and checked: decay rate within 2.6 % of the locked-shaft eigenvalue; self-excitation threshold within 1.2 %.
- [x] Torsional interaction: each mode peaks where `f0 − f_er` meets its frequency; benchmark compensation destabilises mode 2 at +0.358 1/s.
- [x] Time-domain growth rates within 6e-7 of eigenvalues on ten cases. RK4 order 4.00.
- [x] Assumption ledger started.
- [ ] Benchmark generator and network data checked against the 1977 paper (entered from memory; shaft data are confirmed indirectly).

Finding that shaped the plan: on System A the generator always sees negative slip, but the resulting negative resistance is at most 0.016 pu against 0.03 pu of network resistance. Slip alone does not destabilise a large synchronous machine. Phase 2 therefore starts with an induction generator, where it does.

### Phase 2: Induction and doubly-fed machines, validation ladder (3.5 weeks)

1. **System B1 (done, `docs/steps/02-induction-generator.md`).** Induction generator model with stator transients, single mass. Slip calculation for a round rotor. Self-excitation boundary mapped against rotor resistance and against rotor speed, from the hand calculation and from the solver. They agree within 0.9 points of compensation with the rotor free and to 2e-11 with the speed held.
2. **Energy accounting (done, same step).** Energy the machine delivers through the oscillation, from the waveform, against `I² · R_machine`: 1.4 % for a sustained oscillation, at most 2.1 % for growing and decaying ones once the machine is evaluated at the complex frequency.
3. **System B2.** First stage done (`docs/steps/03-dfig-current-control.md`): rotor-side current controller with an ideal frame and held speed. The gain enters the slip calculation as rotor resistance, exactly when `Ki` is kept and within 2 % at standard tuning when it is dropped; the approximation fails as `Ki / Kp` grows (19 % at ten times the tuned ratio). The energy budget shows the converter absorbing energy and the shaft supplying all of it. Second stage done (`docs/steps/04-dfig-full-controls.md`): PLL, outer loops, grid-side converter, DC link and free rotor speed, 23 states. With terminal-voltage feed-forward on the grid-side converter the hand calculation stays within 4 points of compensation of the full solver; without it the grid-side converter damps the mode and the boundary moves by up to 32 points. Published parameter tables could not be retrieved, so the comparison with published values is still open, and the added controls do not explain why the model is unstable at gains well below practice. Step 05 (`docs/steps/05-network-and-impedance.md`) traced most of that gap to the network: the radial connection with no plant impedance is the worst case, and plant impedance plus one parallel path raises the tolerable current-loop bandwidth at rated speed from 3.5 to 19.2 Hz.
4. Validation ladder:
   1. Hand calculations (slip, negative resistance, energy balance).
   2. Internal consistency: eigenvalues against time-domain growth.
   3. Published eigenvalue trends for the doubly-fed machine against compensation and wind speed.
   4. Independent formulation: an open-source electromagnetic-transient tool if one fits without heavy adaptation, otherwise a second implementation in the abc frame with trapezoidal companion models. An impedance scan of the linearised model is run here as one more cross-check of the stability boundary.
   5. Recorded events: digitised waveforms from published reports, compared on frequency and growth rate only.
5. Assumption studies: time step; trapezoidal against RK4; aggregation of N turbines into one machine; and the effect of dropping stator transients, which is what positive-sequence tools do. Time step and dropped transients done (`docs/steps/06-assumption-studies.md`): without network transients the sub-synchronous mode does not exist and every unstable case is reported stable; RK4 at 0.5 ms is accurate to 3e-6 1/s. Rung 4 second implementation done for the current-loop model (`docs/steps/07-phase-coordinates.md`): a phase-coordinate trapezoidal model gives the same growth rates within 0.00014 1/s.
6. `docs/VALIDATION.md` (started in step 05; lists what is done and what is open on every rung).

Exit check: hand calculation and solver agree on the System B1 self-excitation boundary within 2 % of compensation level (met: 0.9 points); energy balance closes within 5 %, at the real frequency for a sustained oscillation and at the complex frequency otherwise (met: 1.4 % and 2.1 %); published System B2 trends reproduced; every larger disagreement has a written cause or is listed as open.

Finding from System B1 that shapes the rest of the phase: the base single-cage machine self-excites only above 96.5 % compensation, not at realistic levels as this plan first assumed. The boundary drops to 73 % with rotor resistance of 0.010 pu and to 47 % with the rotor at 0.70 pu speed. Low speed and added effective rotor resistance are what the doubly-fed machine and its current controller bring, so System B2 is where the mechanism becomes a practical problem.

Known weak point: rung 5. Public recordings come without plant parameters.

### Phase 3: Data and physics features (1.5 weeks)

1. Parameter ranges in a versioned config, each with its source.
2. Sampling: scrambled Sobol plus boundary-focused sampling by bisection toward `σ = 0`.
3. For every case, store the solver outputs **and** the mechanism quantities from the hand calculations. The same file serves arms N, F, R and L.
4. Splits fixed before any training: in-distribution train / validation / test, and extrapolation tests (compensation above the ceiling; short-circuit ratio below the floor; wind speed outside the range; controller gains outside the range; a different machine; inside against outside the validity range of the hand calculation; System C).
5. Generators: eigenvalue-based on CPU processes; trajectories on the GPU in batches of 30,000 or more.
6. Manifest per dataset (config hash, commit, solver version, counts, failures). A 1,000-row sample committed for tests.

Planned sizes: about 200,000 cases for T1, about 30,000 trajectories for T2.

Exit check: datasets regenerate bit-for-bit; splits committed; dataset report written.

Status: the T1 dataset is done (`docs/steps/08-t1-dataset.md`): 131,072 in-distribution cases over 12 parameters, a boundary set of 4,931 and five extrapolation sets of 4,096, bit-identical on regeneration. Ranges and splits live in `src/sso_sciml/data/t1.py`, not in a separate config file. The T2 trajectory dataset is not built yet. No short-circuit-ratio split and no System C.

### Phase 4: T1 with arms N, F and R (2 weeks)

1. Training harness: one config per run, seeded, results in `docs/results/` as JSON with plots of training, performance and metrics (decided 2026-11-11 in place of an MLflow store).
2. Baselines: gradient-boosted trees and Gaussian process, with and without physics features.
3. MLP and deep ensemble under arms N, F and R, at four data budgets.
4. Feature ablation: which mechanism quantity carries the gain.
5. Feature corruption test.
6. Uncertainty: ensemble spread, MC dropout, conformal intervals.
7. Metrics: growth-rate error in 1/s overall and near the boundary; missed-unstable rate; mechanism-consistency rate; inference time.

Exit check: results table for every model, arm, budget and split; verdicts on hypotheses 1 and 2.

Status: gradient-boosted trees and a multilayer perceptron are trained under arms N, F and R at four budgets and three seeds, and scored on every split (`docs/steps/09-t1-arms.md`). Arm R is the best perceptron from 10,000 cases up and on every extrapolation set. Feature ablation, corruption and validity studies are done (`docs/steps/10-t1-feature-studies.md`): the impedance root alone carries the gain of arm F; arm R passes any error of its baseline that is not a function of the inputs straight into the prediction, and arm F learns to discount one. Uncertainty, the Gaussian process, trend consistency and timing are done (`docs/steps/11-t1-uncertainty.md`): ensemble spread is too narrow by a factor of 2 to 4; conformal intervals hold at 90 % in distribution and at 36 to 94 % on the extrapolation sets; flagging the 1 % of cases whose interval contains zero cuts missed unstable cases from 2.0 % to 0.15 % for arm N. Arms F and R are limited by the 1.7 ms hand calculation, 6 times faster than the solver; arm N is about 7,500 times faster in batches. Training curves, parity plots, a grid of every metric and reliability diagrams are drawn by the same scripts. Not done: the trend in rotor speed.

### Phase 5: PINNs (3 weeks)

1. Forward PINN on one System A case, then one System B1 case. Work through the difficulty table; record what each remedy changes.
2. Eigen-residual PINN for T1 (arm L): predict `λ̂, v̂` from θ, trained on `‖A(θ)v̂ − λ̂v̂‖` with zero, few, and many labels.
3. Parametric PINN for T2 over compensation and rotor speed.
4. Inverse PINN on synthetic measurements with known answers.
5. Cost accounting: GPU-hours to reach a given error, against a data-trained network including the cost of generating its data.

Exit check: for each of the four items, either a model that meets a stated error or a documented failure with the evidence for its cause.

Status: item 1 is done for System B1 (`docs/steps/12-forward-pinn.md`), not yet for System A. A single network over 0.5 s settles onto the steady state whatever inputs, activations or causal weighting it has. Scaling each state and marching in windows of 0.05 s are what make it work: 4 % error after 365 s on the GPU, against 4.7 s for RK4. Item 2 is done (`docs/steps/13-eigen-residual.md`) with a negative result: with no labels the eigen-residual finds an eigenvalue of the right matrix but not the sub-synchronous one; with 100 labels a penalty on the smallest singular value of `A − λI` cuts the error from 0.83 to 0.59 1/s at one weight and makes it worse at ten times that weight; with 1,000 labels it changes nothing. The state matrix costs 96 % of a full solve, so arm L is dropped for T1 and kept for the time-domain task only.

### Standing question: the most robust deterministic screen (added 2026-11-12)

Every step from 14 on also answers: what is the most robust deterministic screening approach, and how little information does it need? What does the network topology say alone, and what can a phasor-domain model give? Learned models are judged against a physics rule with a margin, on cases cleared with no unstable case among them, in and out of distribution.

Status (`docs/steps/14-screening-ladder.md`): the reference screen is the root of the loop impedance with a margin of 1 1/s where the PLL is slower than the mode and 5 1/s where it is faster. It clears 79.9 % of cases in distribution with none unstable. A network frequency scan with resistance clears 24.8 % with no plant data. A phasor-domain dynamic model reports every unstable case as stable. For the time-domain task (`docs/steps/15-linear-vs-time-domain.md`): the oscillation is the sum of the modes to within 0.44 % after a 0.2 pu disturbance and parts from it only near 1 pu, so trajectories of the present model need no time stepping and no learned surrogate. The model has no converter limits, so task T2 as written (growth and the limit cycle once limits act) is postponed until limits are added. With a rotor-voltage limit added (`docs/steps/16-converter-limit.md`) there is no limit cycle: the DC-link voltage leaves a 0.2 pu band before the rotor voltage reaches its ceiling in 96 % of cases, and control is lost soon after. The time to the first limit follows from the modes to within 8 %, on the late side. T2 is redefined as "which limit first, and when", which needs no trajectory dataset on this model. The project owner chose the second plant type first, and widened it: any converter, source or load, and not the phase-locked loop alone.

Status for System C (`docs/steps/17-generic-converter.md`): one converter model covers sources and loads, with a phase-locked loop or without. The step 14 rule does not carry over, because a current loop alone is passive here; the negative conductance comes from the loop that holds power constant (phase-locked loop for a source, DC voltage loop for a load). The reference screen is on 2 × 2 scans of converter admittance and network impedance: loop gain below 1 wherever the converter is not passive. Its threshold is fixed by theory. It clears 76 % of cases with a phase-locked loop and 61 % without, and no unstable case in 60,124, in or outside the training range. A Nyquist count on the same scans is exact when the scan is taken at the power-flow operating point. Positive-sequence scans miss 40 % of unstable cases or more. Open: a source and a load on one bus; control delay; modelling the protection (DC chopper, crowbar, anti-windup, trip) to study what follows the first limit.

Status for both plant types (`docs/steps/18-one-screen-two-plants.md`): the doubly-fed plant was scanned as a 2 × 2 admittance. The loop-gain test of step 17 is safe on it and clears 0.2 %, because a series resonance has a loop gain near 1 whether stable or not. The reference screen for both plant types is now a gain margin with a 30° guard on the eigenvalues of the loop matrix, then a Nyquist count. The first clears 36 %, 86 % and 74 % of cases (doubly-fed, converter with and without a phase-locked loop) and no unstable case in 98,635; the second agrees with the eigenvalues in all but 1 of 93,704. Scans must be fine: a thin scan clears unstable cases. Stability is scored on any mode from here on; the step 14 rule, scored that way, clears 478 test cases unstable in a slow outer-loop mode. Open: whether the slow outer-loop mode survives a realistic tuning.

Status for the rational fit (`docs/steps/19-rational-fit.md`): a rational function fitted to `det(I + Y Z)` on a clean scan of 251 points gives the right verdict in all 98,635 cases and the growth rate to within 0.01 1/s in 99.9 % of them; the number of points needed follows the order of the system, not the sharpness of its resonances. It is the reference screen for scans taken from a model. With 5 % noise it fails, and the guarded gain margin on a dense scan stays the reference for noisy scans. A single margin in 1/s is set by the slow mode of the doubly-fed plant (0.2 to 0.4 1/s) and is not adopted. Both taken up in step 20.

Status for noisy scans (`docs/steps/20-noise-and-modes.md`): the interpolating fit fails at a noise of one part in a million and is kept for exact scans only. The reference for any other scan is a least-squares fit of the admittance (vector fitting, order from the noise level, states cut to what the scan supports), closed exactly on the known network, with modes sorted into bands below 2 Hz, 2 to 58 Hz and above. In the sub-synchronous band the growth-rate error is about 0.06 1/s per percent of scan error in the median. A margin of 0.5 1/s is safe up to 1 % noise and 5 1/s at 5 %. Quadrupling the points helps little. The slow mode of the doubly-fed plant is too weakly seen at the terminals to screen from a noisy scan. Open: error in the network data; a fit whose order grows with the data.

Status for the slow mode (`docs/steps/21-slow-mode.md`): it is a mode of the rotor current controller integrators, the outer loops and rotor speed, present in every case near 0.5 Hz and unstable in 3.6 % of the test fold, where the current loop is below about 8 Hz, the outer loops are fast and the rotor speed is high. The series capacitor plays no part. The phasor-domain model reproduces it to 0.006 1/s at the 99th percentile; the plant alone on an ideal voltage misses 96 % of the unstable cases. So the phasor-domain model, which sees no resonance, is the screen for the band below 2 Hz. For the doubly-fed plant the step 14 rule and the phasor-domain model together clear 75.6 % of test cases with no unstable case, using no scan and no time-domain run. Open: a source and a load on one bus; control delay.

Status for network data error (`docs/steps/22-network-error.md`): with the plant exact and each of five network quantities wrong by an independent factor, the growth rate of the resonance is off by about 0.08 1/s per percent in the median and four times that at the 95th percentile, the same size as the effect of scan noise. The series capacitor and line resistance matter most for the doubly-fed plant, line reactance for a converter. Margins of several 1/s are needed; at 10 % error only 10 1/s clears no unstable case. The step 14 rule absorbs 2 % with its own margins. The slow mode moves by less than 0.001 1/s for a 10 % error, so the phasor-domain screen is robust. Open: structured error from one outage.

Status for uncertain network data (`docs/steps/23-network-corners.md`): with each of five network quantities known to within a bound, the plant model is closed on the network at all 32 ends of the ranges and the largest growth rate decides, with no margin. It clears no unstable case at bounds of 5 %, 10 % and 20 % on either plant type, and at 10 % it clears 56 % of doubly-fed cases and 97 % and 77 % of converters, where a margin as safe (10 1/s) clears 10 %, 2 % and 3 %. The single end that is unfavourable in the median is the worst end in only about 60 % of cases and is not used alone. The worst end is not a strict bound (7 cases of 9,630 exceed it, none cleared wrongly). Open: one outage at a time; a source and a load on one bus; control delay.

Status for both error sources together (`docs/steps/24-noise-and-ranges.md`): a noisy 126-point scan fitted by least squares and closed on the 32 ends of the network ranges clears no unstable case in 24,526 with 1 % scan noise and the network within 5 % or 10 %, with no margin. It clears 56 %, 96.4 % and 76.8 % of cases (doubly-fed, converter with and without a phase-locked loop) against 72.4 %, 97.6 % and 82.7 % that are stable. With 5 % noise a margin of 1 1/s on top is needed. This is the reference screen: plant to 1 % at a hundred frequencies, network to 10 %.

Status for outages (`docs/steps/25-outages.md`): four outages were applied to every case stable on the intact network. Losing the parallel path is the worst for every plant type (32 % of doubly-fed cases become insecure, 13 % and 17 % of converters). A margin on the intact network does not cover it: about one case in ten with a margin of 10 1/s is insecure afterwards. Half the units out makes 10 % of doubly-fed cases unstable and helps a converter. For the doubly-fed plant the plant model taken before the outage gives the growth rate after it to 0.01 1/s, so one scan serves every outage; for a converter it does not, because the admittance moves with the operating point. Each credible outage is a separate case for the screen. Open: a source and a load on one bus; control delay.

Status for the operating point (`docs/steps/26-operating-point.md`): at a given power the admittance of the generic converter depends on one number, the reactive current with a phase-locked loop and the terminal voltage without. A scan retaken at its value after an outage is exact. With a phase-locked loop two scans 0.1 pu apart and a straight line give no wrong verdict in 43,942 outage cases; without one three scans and a parabola clear 2 unstable cases of 695, where one reused scan clears 147. The power flow after each outage supplies the number. Open: the admittance along the power axis; a source and a load on one bus; control delay and a filter capacitor.

### Phase 6: The loss-against-feature comparison (2 weeks)

1. Arms N, F, R, L, A and F+L on T1 and T2, matched architectures, four data budgets, five seeds.
2. All extrapolation splits.
3. Sensitivity: loss weight for L; feature error for F.
4. Statistical treatment: paired comparisons across seeds, intervals on every difference.
5. Report with a verdict on each of the five hypotheses.

Exit check: one table and one figure per task that a reader can use to choose an arm for a given data budget and extrapolation need.

### Phase 7: T2 operators and sequence models (2 weeks)

1. Disturbances: compensation step, voltage dip, wind ramp.
2. DeepONet, Fourier neural operator, temporal convolutional network, transformer; the neural ODE from arm A.
3. Rollout evaluation: error against horizon, growth-rate error, limit-cycle amplitude error, false divergence rate.
4. Memory and time profile of each on the 8 GB card.

Exit check: results table; the horizon beyond which each model should not be trusted.

### Phase 8: T3 measurement inverse (1.5 weeks)

1. Measurement model: noise, PMU filtering and frame rate (which aliases sub-synchronous modes), fault-recorder sampling.
2. Matrix pencil and Prony; 1-D CNN; identification through the differentiable solver; inverse PINN from Phase 5.
3. Identifiability: which parameters a terminal waveform constrains.
4. Applied to the digitised event waveforms from Phase 2.

Exit check: error against signal-to-noise ratio per method; a list of parameters that cannot be identified.

### Phase 9: Out-of-distribution study and fallback (2 weeks)

1. Every model on every extrapolation split; error against distance outside the range, per direction.
2. Detectors: ensemble disagreement, conformal width, input-space and latent-space distance, physics residual of the prediction (available for free in arms L and A), and disagreement between the network and the hand calculation.
3. Score detectors by how well they rank cases by true error and by missed-unstable rate at a fixed solver budget.
4. Fallback rule tuned on validation, reported on test, with the trade-off curve.

Exit check: a policy with a stated operating point (target: no missed unstable case on the in-distribution test set with the solver called on under 10 % of cases) and a list of directions where the model fails without warning.

### Phase 10: Design search (1.5 weeks)

Choose rotor-side current-loop gains, a damping-controller gain and filter frequency, and the number of compensation segments in service (integer), to maximise worst-case damping over a scenario set, subject to a control bandwidth floor.

1. Random search, CMA-ES, Gaussian-process Bayesian optimisation on the solver, and optimisation over the surrogate with solver confirmation.
2. Compare on solver calls to reach a given worst-case damping.
3. Confirm each optimum in the nonlinear time-domain solver.

Exit check: table of best design, solver calls and wall time per method.

### Phase 11: Service and release (1.5 weeks)

1. Model registry: version, dataset hash, training commit, metrics, valid input ranges.
2. FastAPI service. `POST /screen` returns growth rate and frequency with an interval, the mechanism quantities (slip at resonance, `R_machine`, `R_net`) as the explanation, an out-of-range flag, and `source: surrogate | solver`. `POST /simulate` returns a trajectory.
3. Tests: golden numbers (the Phase 1 values are already pinned), analytical limits, monotonic trends, and the contract that out-of-range input returns `source: solver`.
4. Model cards. Tagged release.

Exit check: fresh clone, `uv sync`, one command starts the service and the suite passes.

### Phase 12: Report (1 week)

Cross-model comparison, validation record, failure catalogue, verdicts on the hypotheses, and what would be done differently.

### Stretch goals

- System C in full.
- Multi-plant network with a graph neural network that takes topology as input.
- Active learning driven by surrogate uncertainty.

### Schedule

About 22 part-time weeks remain for Phases 2 to 12. Phases 7 and 8 do not depend on each other. The minimum complete path is Phases 2 to 6, 9 and 11 on T1 alone: validated solver, the physics-placement comparison, an out-of-distribution study, and a service.

## 7. Documentation

| File | Contents |
|---|---|
| `docs/PLAN.md` | This plan, with a dated note for each revision |
| `docs/steps/NN-name.md` | One entry per completed step: goal, commands, results, disagreements, decisions, next |
| `docs/EQUATIONS.md` | Model equations and conventions |
| `docs/ASSUMPTIONS.md` | Assumption ledger |
| `docs/VALIDATION.md` | Every solver comparison and its outcome (from Phase 2) |
| `docs/results/` | Tables and figures, each produced by a script in the repo |
| `docs/model_cards/` | One card per released model |
| `docs/DECISIONS.md` | Short dated records of choices that were not obvious |

Rule: a number in a document must be reproducible from a command recorded next to it.

## 8. Repository layout

```
sso-sciml/
├── configs/            parameter ranges, dataset and training configs
├── data/               generated, not committed (samples/ is committed)
├── docs/
├── scripts/            runnable entry points
├── src/sso_sciml/
│   ├── physics/        machines, shaft, network, slip and damping-torque calculations
│   ├── solver/         linearisation, integration, modal estimation
│   ├── data/           sampling, generation, storage, splits
│   ├── models/         architectures, including PINNs
│   ├── train/          harness, physics losses, uncertainty
│   ├── ood/            detectors and fallback policy
│   ├── design/         optimisation
│   └── service/        API and registry
└── tests/
```

`physics/` and `solver/` exist. The rest are added by the phase that needs them, as are dependencies (`xgboost`, `gpytorch`, `botorch`, `torchdiffeq`, `zarr`, `fastapi`, `pydantic`, `hypothesis`).

## 9. Where a learned model will not beat the physics

Stated in advance.

- For one machine with a known model, the hand calculation takes microseconds and the eigenvalue solve takes milliseconds. A network does not beat either on speed. Phase 1 already showed the hand calculation within 2.6 % of the solver for mechanism 1. The case for a learned model rests on three things: plants whose internals are not available so only time-domain runs exist; nonlinear behaviour after limits act; and design searches needing very many evaluations. Speed is reported against the hand calculation and the eigenvalue solve as well as the time-domain run.
- A forward PINN for a single case is slower than RK4 by orders of magnitude. It is in the plan as a stepping stone, not a product.
- The reference solver stands in for a vendor model. Agreement between surrogate and solver says nothing about agreement between solver and a real plant. That gap is covered only by Phase 2.
- A model trained on one mechanism is not expected to transfer to another. Phase 9 measures how badly.

## 10. Standards and practice referenced

To be read and cited properly in Phase 2.

- IEEE Std 2800-2022, interconnection requirements for inverter-based resources.
- NERC reliability guidelines and event reports on inverter-based resource performance and sub-synchronous oscillation.
- ERCOT Nodal Protocols section on subsynchronous resonance study requirements.
- IEEE PES technical report on wind energy systems sub-synchronous oscillations.
- IEEE SSR working group benchmark model papers.

## 11. Starting references

Citations to be confirmed with DOIs.

- IEEE SSR Task Force, "First benchmark model for computer simulation of subsynchronous resonance," IEEE Trans. Power Apparatus and Systems, vol. 96, no. 5, 1977. DOI 10.1109/T-PAS.1977.32485.
- P. Kundur, Power System Stability and Control, 1994 (machine equations, chapter on sub-synchronous oscillations).
- P. Anderson, B. Agrawal, J. Van Ness, Subsynchronous Resonance in Power Systems, 1990.
- L. Fan, R. Kavasseri, Z. Miao, C. Zhu, "Modeling of DFIG-based wind farms for SSR analysis," IEEE Trans. Power Delivery, 2010.
- A. Ostadi, A. Yazdani, R. Varma, "Modeling and stability analysis of a DFIG-based wind-power generator interfaced with a series-compensated line," IEEE Trans. Power Delivery, 2009.
- J. Adams, C. Carter, S.-H. Huang, "ERCOT experience with sub-synchronous control interaction and proposed remediation," IEEE PES T&D, 2012.
- Y. Cheng et al., "Real-world subsynchronous oscillation events in power grids with high penetrations of inverter-based resources," IEEE Trans. Power Systems, 2023.
- M. Raissi, P. Perdikaris, G. Karniadakis, "Physics-informed neural networks," J. Computational Physics, 2019.
- S. Wang, Y. Teng, P. Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks," SIAM J. Scientific Computing, 2021.
- S. Wang, S. Sankaran, P. Perdikaris, "Respecting causality is all you need for training physics-informed neural networks," 2022.
- M. Tancik et al., "Fourier features let networks learn high frequency functions in low dimensional domains," NeurIPS, 2020.
- C. Rackauckas et al., "Universal differential equations for scientific machine learning," 2020.
- L. Lu et al., "Learning nonlinear operators via DeepONet," Nature Machine Intelligence, 2021.
- Z. Li et al., "Fourier neural operator for parametric partial differential equations," ICLR, 2021.
- R. Chen et al., "Neural ordinary differential equations," NeurIPS, 2018.
- B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," NeurIPS, 2017.
- A. Angelopoulos, S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," 2021.

## 12. Risks

| Risk | Effect | Response |
|---|---|---|
| Doubly-fed machine model does not reproduce published trends | Phase 2 stalls | System A proved the network and solver; System B1 isolates the induction machine before controls are added |
| No independent tool models System B2 without major work | Validation rung 4 weakens | Second implementation in the abc frame |
| No usable public measured waveforms | Rung 5 reduces to frequency and growth rate from published figures | Already assumed and stated |
| PINNs do not train on oscillatory, stiff trajectories | Phase 5 yields a negative result | Planned for: the difficulty table lists remedies in order; a documented failure meets the exit check |
| Arm comparison is confounded by tuning effort | Phase 6 conclusions unreliable | Fixed tuning budget per arm, stated in advance; five seeds; paired statistics |
| Hand-calculation features leak the answer | Arm F looks better than it is | By design the feature is cheap physics, not solver output. Report arm F against the hand calculation alone as a baseline |
| Mode switching makes T1 targets discontinuous | Poor accuracy near switches | Track modes by participation; predict each tracked mode |
| 8 GB limits time-domain models and PINN collocation batches | Smaller models or windows | Mixed precision where residual precision allows, windowed training; report the limit |
| Scope is large for one person | Later phases slip | Minimum complete path defined in the schedule |
