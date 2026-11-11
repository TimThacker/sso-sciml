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

**System B1: fixed-speed induction generator on a series-compensated line (Phase 2).** The cleanest case of mechanism 1. Rotor resistance is much higher than in a synchronous machine and the slip calculation is exact for a round rotor, so the hand calculation and the solver should agree closely and self-excitation should appear at realistic compensation.

**System B2: doubly-fed induction generator with converter controls (Phase 2, main study).** Mechanism 3. About 24 states. Parameter space of about 14 dimensions:

| Group | Parameters |
|---|---|
| Network | compensation level, line R/X, line length, grid short-circuit ratio |
| Operating point | wind speed (sets rotor speed, hence slip), turbines online, terminal voltage, reactive dispatch |
| Controls | rotor-side current loop Kp and Ki, outer loop bandwidth, PLL bandwidth, grid-side current loop Kp |
| Machine | rotor resistance, leakage reactances |

Trends the solver must reproduce before data generation: damping falls as compensation rises, as wind speed drops (slip at resonance moves toward zero from below, so `R_r / s` grows), and as rotor-side current loop gain rises.

**System C: grid-following converter on a weak grid (extension).** No rotor and no slip; the oscillation comes from PLL and grid-impedance interaction. Used in Phase 9 as a deliberately different mechanism to test transfer.

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

1. **System B1.** Induction generator model with stator transients, single-mass or two-mass drive train. Slip calculation for a round rotor. Map the self-excitation boundary in the plane of compensation level against rotor speed, from the hand calculation and from the solver, and overlay them.
2. **Energy accounting.** From the time-domain run, compute the energy the machine delivers at the sub-synchronous frequency per cycle and confirm that it matches `I² · R_machine`. This ties the waveform to the mechanism.
3. **System B2.** Rotor-side and grid-side converter controls, DC link, PLL. Extend the slip calculation with the current-loop gain as an effective rotor resistance, and measure where that approximation fails.
4. Validation ladder:
   1. Hand calculations (slip, negative resistance, energy balance).
   2. Internal consistency: eigenvalues against time-domain growth.
   3. Published eigenvalue trends for the doubly-fed machine against compensation and wind speed.
   4. Independent formulation: an open-source electromagnetic-transient tool if one fits without heavy adaptation, otherwise a second implementation in the abc frame with trapezoidal companion models. An impedance scan of the linearised model is run here as one more cross-check of the stability boundary.
   5. Recorded events: digitised waveforms from published reports, compared on frequency and growth rate only.
5. Assumption studies: time step; trapezoidal against RK4; aggregation of N turbines into one machine; and the effect of dropping stator transients, which is what positive-sequence tools do.
6. `docs/VALIDATION.md`.

Exit check: hand calculation and solver agree on the System B1 self-excitation boundary within 2 % of compensation level; energy balance closes within 5 %; published System B2 trends reproduced; every larger disagreement has a written cause or is listed as open.

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

### Phase 4: T1 with arms N, F and R (2 weeks)

1. Training harness: one config per run, seeded, local MLflow store, results tables in `docs/results/`.
2. Baselines: gradient-boosted trees and Gaussian process, with and without physics features.
3. MLP and deep ensemble under arms N, F and R, at four data budgets.
4. Feature ablation: which mechanism quantity carries the gain.
5. Feature corruption test.
6. Uncertainty: ensemble spread, MC dropout, conformal intervals.
7. Metrics: growth-rate error in 1/s overall and near the boundary; missed-unstable rate; mechanism-consistency rate; inference time.

Exit check: results table for every model, arm, budget and split; verdicts on hypotheses 1 and 2.

### Phase 5: PINNs (3 weeks)

1. Forward PINN on one System A case, then one System B1 case. Work through the difficulty table; record what each remedy changes.
2. Eigen-residual PINN for T1 (arm L): predict `λ̂, v̂` from θ, trained on `‖A(θ)v̂ − λ̂v̂‖` with zero, few, and many labels.
3. Parametric PINN for T2 over compensation and rotor speed.
4. Inverse PINN on synthetic measurements with known answers.
5. Cost accounting: GPU-hours to reach a given error, against a data-trained network including the cost of generating its data.

Exit check: for each of the four items, either a model that meets a stated error or a documented failure with the evidence for its cause.

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

`physics/` and `solver/` exist. The rest are added by the phase that needs them, as are dependencies (`xgboost`, `gpytorch`, `botorch`, `torchdiffeq`, `mlflow`, `zarr`, `fastapi`, `pydantic`, `hypothesis`).

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
