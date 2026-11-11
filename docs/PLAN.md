# Project plan

Status: Phase 0 complete (2026-11-11). Nothing below Phase 0 has been run yet; every target number in this document is a target, not a result.

## 1. What this project is

Sub-synchronous oscillation (SSO) screening for series-compensated and inverter-rich grids, using learned models next to a physics solver.

An SSO study today means electromagnetic-transient (EMT) simulation and impedance scans, repeated for every combination of operating point, contingency, and controller setting. Each scan costs minutes. Vendor converter models are usually black boxes, so there is no state matrix to take eigenvalues of, and the scan is the only tool. A study covers a few hundred cases out of a space with millions.

This project builds:

1. A reference solver written from first principles, with the network electromagnetic dynamics kept (no quasi-static phasor assumption), validated against published benchmarks, an independent formulation, and recorded events.
2. Several kinds of learned surrogate trained on that solver, compared on the same data and the same splits.
3. A measured account of where each surrogate fails outside its training distribution, how the failure is detected, and when the system falls back to the solver.
4. A design search over mitigation settings that uses the surrogate for speed and the solver for confirmation.
5. A versioned service with tests that fail when a number is wrong.

### Why SSO rather than GIC

Both were candidates. SSO was chosen for three reasons.

- **The solver is worth replacing.** A GIC study is a DC network solve, one sparse linear system per storm scenario. A surrogate has little to beat. An SSO scan is a time-domain EMT run per frequency point per operating point, so the cost gap a surrogate can close is two to four orders of magnitude.
- **It covers more of the target scope.** SSO sits at the intersection of converter control dynamics, network electromagnetics, and interconnection study practice. GIC touches transformer saturation and geophysics, which is a narrower slice of the electrical lane.
- **There is published ground truth.** The IEEE First Benchmark Model and the DFIG series-compensation literature give eigenvalues and mode frequencies to check the solver against before any ML is trained.

### Scope boundaries

In scope: balanced three-phase dynamics in the dq frame, 0.1 to 120 Hz, small-signal and large-signal. Out of scope for now: unbalanced faults, switching-level converter models, transformer saturation, protection logic. Each exclusion is revisited in Phase 2 when the solver is compared with an independent formulation.

## 2. How the plan maps to the scope document

| Scope item | Where it is addressed |
|---|---|
| Physics models you set up yourself; know which assumptions and discretization choices drive the answer | Phases 1 and 2. Assumption ledger in `docs/ASSUMPTIONS.md`; step-size and frame studies in Phase 2 |
| Checked against experiments or measured data; say what you did when they disagreed | Phase 2 validation ladder; every step log has a "Disagreements" section |
| Surrogates, neural operators, physics-informed networks | Phases 4 to 7 |
| Out-of-distribution behaviour, detection, solver fallback | Phase 8, with extrapolation splits fixed in Phase 3 before any model is trained |
| Decide where solver fidelity is required and where a learned model is enough | Phase 8 fallback policy; Section 9 lists where the surrogate is not worth using |
| Search a design space; Bayesian, mixed-integer, derivative-free | Phase 9 |
| Something they can run; behind an API, versioned, tests that catch a wrong number | Phase 10 |
| Training runs designed and debugged in PyTorch; uncertainty quantification | Phases 4 to 7; UQ in Phases 4 and 8 |
| Standards the domain works to | Section 10 |

## 3. Physical systems

Three systems, in order of increasing novelty. The first exists to prove the solver. The second is the main study. The third is an extension.

### System A: IEEE First Benchmark Model (validation anchor)

A synchronous generator with a six-mass turbine shaft (HP, IP, LPA, LPB, generator, exciter) feeding an infinite bus through a series-compensated line. This is classical sub-synchronous resonance: the electrical resonance, seen from the rotor, lines up with a torsional mode.

Used for: checking the dq network formulation, the shaft model, and the eigenvalue and time-domain machinery against published torsional mode frequencies (about 15.7, 20.2, 25.6, 32.3, and 47.5 Hz in the literature; to be confirmed against the source paper in Phase 1).

### System B: Type-3 wind plant on a series-compensated line (main study)

An aggregated doubly-fed induction generator (DFIG) plant connected radially to a series-compensated line. This is sub-synchronous control interaction (SSCI), the mechanism behind the 2009 South Texas event. At sub-synchronous frequency the rotor slip is negative, so the rotor resistance looks negative from the stator, and the rotor-side converter current loop makes it more negative.

States: stator and rotor flux (4), series capacitor voltage (2), line current (2), DC link (1), rotor-side and grid-side converter controller integrators (about 8), PLL (2), two-mass drive train (3), grid-side filter (2). About 24 states.

Parameter space (the surrogate inputs), about 14 dimensions:

| Group | Parameters |
|---|---|
| Network | series compensation level, line R/X, line length, grid short-circuit ratio |
| Operating point | wind speed (sets rotor speed), turbines online, terminal voltage, reactive dispatch |
| Controls | rotor-side current loop Kp and Ki, outer loop bandwidth, PLL bandwidth, grid-side current loop Kp |
| Machine | rotor resistance, leakage reactances (ranges spanning published machine data) |

Known trends the solver must reproduce before any data is generated: damping falls as compensation rises, falls as wind speed drops, and falls as rotor-side current loop gain rises.

### System C: Grid-following converter on a weak grid (extension)

A full-converter resource or large electronic load with PLL and current control, connected through a weak and optionally compensated grid. Sub-synchronous oscillation here comes from PLL and grid-impedance interaction at low short-circuit ratio. Used in Phase 8 as a test of transfer to a different mechanism, and in the stretch goals.

### Solver formulation

One model definition, written in PyTorch in float64, gives three products:

1. **Nonlinear time-domain simulation.** `dx/dt = f(x, u, θ)`, integrated with a fixed-step method on the GPU for batches and with SciPy's adaptive stiff integrators on the CPU as the reference.
2. **Small-signal model.** Equilibrium by Newton iteration, Jacobian by automatic differentiation (`torch.func.jacrev`), eigenvalues and participation factors on the CPU.
3. **Impedance scan.** The 2×2 dq impedance `Z(jω)` at the plant terminals, two ways: analytically from the linearised model, and by perturbation injection in the time domain, which is the method that works on a black-box model.

Writing the model once in an autodiff framework is what makes the physics-informed losses in Phases 6 and 7 possible: the solver residual is differentiable with respect to both the state and the parameters.

PyTorch rather than JAX because JAX has no GPU support on native Windows, and this machine runs Windows.

## 4. Hardware and compute budget

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3070, 8 GB (about 6.9 GiB free with the desktop running) |
| CPU | Intel i7-10700K, 8 cores / 16 threads |
| RAM | 32 GB |
| Disk budget for data | 20 GB of the 70 GB free |
| Stack | Python 3.12, PyTorch 2.14.1 + CUDA 13.0, managed by `uv` |

Measured in Phase 0 (`docs/steps/00-setup.md`):

- Mixed-precision MLP training runs at about 3.1 million samples per second on the GPU, 27 times the CPU rate.
- Batched non-symmetric eigenvalue solves are about 20 to 60 times **slower** on the GPU than on the CPU for the matrix sizes this project uses.

Consequences:

- Eigenvalue-based data generation runs on the CPU, parallelised over processes.
- The GPU is used for training, inference, and batched fixed-step time-domain integration (to be benchmarked in Phase 3 before committing to it).
- Models are sized for 8 GB: mixed precision by default, batch sizes chosen per model, gradient checkpointing for the neural ODE if needed. `torch.compile` is not assumed to work on Windows.

Estimated GPU time for the whole plan is under 80 hours. No single run is planned to exceed 6 hours.

## 5. Learning tasks and model types

Five tasks. Each has a non-neural baseline, so that a neural model has to earn its place.

| Task | Input → output | Models compared |
|---|---|---|
| **T1 Stability margin** | parameters θ → damping and frequency of the tracked sub-synchronous modes, stable/unstable | Gradient-boosted trees (GPU), Gaussian process (small-N), MLP, deep ensemble of MLPs |
| **T2 Impedance operator** | θ → `Z_dq(jω)` over 1 to 120 Hz | MLP on a fixed frequency grid, DeepONet (branch θ, trunk ω), Fourier neural operator in frequency, rational-function network that predicts poles and residues |
| **T3 Time-domain response** | θ, disturbance → state trajectories after the disturbance | Fourier neural operator in time, neural ODE with a physics residual loss, temporal convolutional network, small transformer |
| **T4 Measurement inverse** | noisy waveform window → mode frequency, damping, and plant parameters | Prony / matrix pencil (baseline), 1-D CNN, gradient-based parameter identification through the differentiable solver |
| **T5 Design** | search over mitigation settings for worst-case damping | Random search, CMA-ES, Bayesian optimisation with a GP, Bayesian optimisation over the surrogate with solver confirmation |

Notes on the choices:

- **T1 is not smooth.** The identity of the least-damped mode changes across the parameter space. Predicting the minimum damping directly gives a kinked target. The plan is to track modes by participation and predict each tracked mode, then take the minimum. Both formulations are trained and compared.
- **The rational-function network in T2 is the physics-structured model.** An impedance of a lumped system is a rational function of `s`. A network that outputs poles and residues can only produce physically shaped spectra, and its poles are eigenvalue predictions that can be checked against T1. The question is whether that structure helps or hurts outside the training range.
- **T3 is where linear analysis stops.** Once current limits saturate, a growing oscillation settles into a limit cycle whose amplitude no eigenvalue predicts. T3 models are scored on growth rate, limit-cycle amplitude, and long-rollout stability, not just pointwise error.
- **T4 connects to measured data.** It also exposes a measurement problem: a 30 or 60 frame-per-second PMU stream aliases sub-synchronous modes. The study includes both PMU-rate and fault-recorder-rate inputs.

Uncertainty quantification, applied across T1 to T3: deep ensembles, MC dropout, and split conformal prediction for intervals with a coverage guarantee on exchangeable data. The GP provides its own posterior. Calibration is reported for each.

## 6. Phases

Every phase ends with a step log in `docs/steps/`, a commit, and the exit check listed. Effort is in part-time weeks for one person.

### Phase 0: Setup (done)

- [x] Public repository created following the dev guide: `uv` project, pinned Python, lockfile, `.gitignore`, first commit, GitHub remote.
- [x] CUDA build of PyTorch installed and verified on the RTX 3070.
- [x] GPU benchmark script for the two workloads the plan depends on.
- [x] CI running lint and CPU tests.
- [x] This plan.

### Phase 1: Solver core and System A (2 weeks)

1. Write `docs/EQUATIONS.md`: per-unit system, dq transformation convention, RLC network in the rotating frame, synchronous machine, multi-mass shaft. State every sign convention.
2. Implement the model interface: `f(x, u, θ)`, equilibrium solve, linearisation, eigenvalues, participation factors.
3. Implement time-domain integration: fixed-step RK4 and trapezoidal in PyTorch, SciPy Radau and LSODA wrappers as reference.
4. Implement System A.
5. Validate:
   - Torsional mode frequencies with the electrical system removed, against the published values.
   - Electrical resonance frequency against `f_er = f0·sqrt(Xc/XL)`, appearing at `f0 − f_er` and `f0 + f_er` in the dq frame.
   - Unstable compensation ranges against published results.
   - Ring-down damping from the time-domain run (matrix pencil fit) against the eigenvalue real part.
6. Start `docs/ASSUMPTIONS.md`: one row per assumption, what it affects, how it was tested.

Exit check: torsional frequencies within 0.1 Hz of published values; time-domain and eigenvalue damping agree within 2 % on ten test cases; all encoded as tests.

### Phase 2: System B and the validation ladder (3 weeks)

1. Implement the DFIG, drive train, DC link, rotor-side and grid-side controls, PLL.
2. Implement both impedance scans (analytical and perturbation injection).
3. Validation ladder, cheapest first:
   1. **Analytical limits.** Controls frozen: the plant impedance reduces to the induction machine equivalent circuit. Compensation zero: no sub-synchronous network mode.
   2. **Three-way internal consistency.** Eigenvalues, time-domain ring-down, and the impedance-based stability criterion must agree on the stability boundary.
   3. **Published results.** Eigenvalue trends against compensation level and wind speed from the DFIG SSR literature.
   4. **Independent formulation.** Either an open-source EMT tool, if one models this system without heavy adaptation (candidates evaluated at the start of this phase), or a second implementation written in the abc frame with trapezoidal companion models. The second option also provides the discretization study.
   5. **Recorded events.** Digitised waveforms from published event reports. The comparison is on oscillation frequency and growth rate only, because site parameters are not public.
4. Discretization and assumption studies: time step against mode frequency and damping error; numerical damping of trapezoidal against backward Euler; effect of the aggregation assumption (N turbines as one machine); effect of neglecting stator transients, which is the assumption positive-sequence tools make and the one expected to remove the phenomenon altogether.
5. Write `docs/VALIDATION.md` with every comparison, the size of each disagreement, and what was done about it.

Exit check: the three internal methods agree on the stability boundary within 1 % of compensation level; published trends reproduced; every disagreement larger than tolerance has a written cause or is listed as open.

Known weak point: rung 5. Public SSO recordings come without the plant parameters. The plan does not claim waveform-level agreement with field data, and the validation document will say so.

### Phase 3: Data generation (1.5 weeks)

1. Parameter ranges fixed in a versioned config, each range with its source.
2. Sampling: scrambled Sobol for coverage, plus boundary-focused sampling (bisection toward the stability boundary), because uniformly sampled data wastes most of its points far from the boundary where the answer is obvious.
3. Splits, fixed and committed **before any model is trained**:
   - In-distribution train / validation / test.
   - Extrapolation tests: compensation above the training ceiling; short-circuit ratio below the training floor; wind speed outside the range; controller gains outside the range; a different machine parameter set; System C.
4. Generators: T1 and T2 on CPU processes; T3 trajectories with batched GPU integration if the Phase 3 benchmark shows it is faster and matches the CPU reference.
5. Storage: one array store per dataset with a manifest (config hash, git commit, solver version, row count, failed-case count). A 1,000-row sample is committed for tests.
6. Dataset report: marginal coverage, fraction unstable, distribution of distance to the boundary, count of solver failures and what caused them.

Planned sizes: about 200,000 cases for T1 and T2, about 30,000 trajectories for T3. Sizes are revised after the learning curves in Phase 4.

Exit check: datasets regenerate bit-for-bit from the config on this machine; splits committed; dataset report written.

### Phase 4: T1 stability-margin surrogates (1.5 weeks)

1. Common training harness: config file per run, seeded, logs to a local MLflow store, checkpoints outside git, results table written to `docs/results/`.
2. Train the four model types. Learning curves at 1k, 10k, 50k, 200k samples.
3. Mode-tracked target against direct minimum-damping target.
4. UQ: ensemble spread, MC dropout, conformal intervals. Report coverage and interval width in and out of distribution.
5. Metrics: damping error in 1/s and frequency error in Hz overall and within a band around the boundary; missed-unstable rate (the costly error); inference time per case.

Exit check: a results table with every model on every split, and a written paragraph on which model would be deployed and why.

### Phase 5: T2 impedance operator surrogates (2 weeks)

1. Output representation study: real/imaginary against log-magnitude/phase; the choice matters near resonance peaks.
2. Train the four model types.
3. Physics checks on predictions: passivity where the true system is passive, conjugate symmetry, agreement between the stability verdict from predicted impedance and the T1 verdict.
4. Resolution test for the operator models: train on one frequency grid, evaluate on a finer one.
5. Speed comparison against both solver scans, reported separately (see Section 9).

Exit check: results table; resonance peak frequency error and peak magnitude error reported separately from average error, since average error hides the part that matters.

### Phase 6: T3 time-domain surrogates (2.5 weeks)

1. Disturbance set: compensation step (line switching), voltage dip, wind ramp.
2. Train the four model types. The neural ODE gets a physics residual term computed from the differentiable solver; an ablation removes it.
3. Rollout evaluation: error against horizon; growth-rate error; limit-cycle amplitude error; fraction of rollouts that diverge when the truth does not.
4. Memory and time profiling on the 8 GB card for each model.

Exit check: results table; a clear statement of the horizon beyond which each model should not be trusted.

### Phase 7: T4 measurement inverse (1.5 weeks)

1. Synthetic measurement model: sensor noise, PMU filtering and frame rate, fault-recorder sampling.
2. Baseline: matrix pencil and Prony on the same windows.
3. 1-D CNN for mode frequency and damping; gradient-based parameter identification through the solver for plant parameters.
4. Identifiability study: which parameters the waveform actually constrains, from the Fisher information of the linearised model.
5. Apply to the digitised event waveforms from Phase 2 and report what is and is not recoverable.

Exit check: error against signal-to-noise ratio for each method; an explicit list of parameters that cannot be identified from terminal measurements.

### Phase 8: Out-of-distribution study and fallback policy (2 weeks)

1. Run every trained model on every extrapolation split. Plot error against distance outside the training range, per direction.
2. Detectors compared: ensemble disagreement, conformal interval width, distance in input space (Mahalanobis, k-nearest-neighbour), distance in latent space, physics-consistency checks from Phase 5.
3. Score each detector by how well it ranks cases by actual error, and by the missed-unstable rate at a fixed solver-call budget.
4. Fallback policy: a rule that sends a case to the solver when a detector fires or when the prediction lies within the uncertainty band of the stability boundary. Tune on validation, report on test.
5. Report the trade-off curve: fraction of cases sent to the solver against missed-unstable rate.

Exit check: a policy with a stated operating point (target: no missed unstable cases on the in-distribution test set and the solver called on under 10 % of cases), and a documented list of the extrapolation directions where the surrogate fails without warning, if any exist.

### Phase 9: Design search (1.5 weeks)

Problem: choose rotor-side current loop gains, a damping-controller gain and filter frequency, and the number of compensation segments in service (integer), to maximise worst-case damping over a scenario set (wind speeds, turbines online, contingencies), subject to a control bandwidth floor.

1. Methods: random search, CMA-ES, GP Bayesian optimisation on the solver directly, and optimisation over the surrogate with every candidate optimum confirmed by the solver.
2. Mixed-integer handling for the segment count.
3. Compare on solver calls needed to reach a given worst-case damping.
4. Check each reported optimum in the nonlinear time-domain solver, not just the eigenvalues.

Exit check: a table of best design, solver calls, and wall time per method; confirmation runs for every reported optimum.

### Phase 10: Service and release (1.5 weeks)

1. Model registry: each model is stored with its version, dataset hash, training commit, metrics, and valid input ranges.
2. FastAPI service. `POST /screen` takes a parameter set and returns the damping and frequency prediction, an interval, an out-of-range flag, and `source: surrogate | solver`. `POST /impedance` returns the spectrum.
3. Tests:
   - Golden numbers: System A torsional modes and a fixed set of System B cases, solver and surrogate.
   - Properties: analytical limits from Phase 2; monotonic trends; conjugate symmetry.
   - Contract: out-of-range input must return `source: solver`.
4. Model cards in `docs/model_cards/`.
5. Tagged release.

Exit check: a fresh clone followed by `uv sync` and one command starts the service on this machine and passes the test suite.

### Phase 11: Report (1 week)

A single document with the cross-model comparison, the validation record, the failure catalogue, and what would be done differently.

### Stretch goals

- System C in full, repeating Phases 3 to 5 and 8.
- Multi-plant network with contingencies and a graph neural network that takes topology as input.
- Active learning: let the surrogate's uncertainty choose the next solver runs, and measure the data saved.

### Schedule

About 21 part-time weeks for Phases 1 to 11 if done in sequence. Phases 5, 6, and 7 are independent of each other once Phase 3 is done, so the order can change. If time is short, the minimum complete path is Phases 1 to 4, 8, and 10 with T1 only: a validated solver, one surrogate task, an out-of-distribution study, and a service.

## 7. Documentation

| File | Contents |
|---|---|
| `docs/PLAN.md` | This plan. Updated when the plan changes, with the reason |
| `docs/steps/NN-name.md` | One entry per completed step: goal, commands, results with units, disagreements, decisions, next step. Template in `docs/steps/TEMPLATE.md` |
| `docs/EQUATIONS.md` | Model equations and conventions |
| `docs/ASSUMPTIONS.md` | Assumption ledger: assumption, what it affects, how tested, result |
| `docs/VALIDATION.md` | Every solver comparison and its outcome |
| `docs/results/` | Results tables and figures, each produced by a script in the repo |
| `docs/model_cards/` | One card per released model |
| `docs/DECISIONS.md` | Short dated records of choices that were not obvious |

Rule: a number in a document must be reproducible from a command recorded next to it.

## 8. Repository layout (target)

```
sso-sciml/
├── configs/            parameter ranges, dataset and training configs
├── data/               generated, not committed (samples/ is committed)
├── docs/
├── scripts/            runnable entry points (check_gpu, generate, train, evaluate)
├── src/sso_sciml/
│   ├── physics/        model definitions, per-unit system, frames
│   ├── solver/         equilibrium, linearisation, integration, impedance scan
│   ├── data/           sampling, generation, storage, splits
│   ├── models/         surrogate architectures
│   ├── train/          training harness, UQ
│   ├── ood/            detectors and fallback policy
│   ├── design/         optimisation
│   └── service/        API and registry
└── tests/
```

Dependencies are added in the phase that first needs them (`uv add`), not up front. Expected additions: `xgboost`, `gpytorch`, `botorch`, `torchdiffeq`, `mlflow`, `zarr`, `fastapi`, `pydantic`, `hypothesis`.

## 9. Where the surrogate will not beat the solver

Stated in advance so the results cannot be framed around it afterwards.

- For a single plant with a known state-space model, eigenvalue analysis takes milliseconds. A surrogate for T1 does not beat it on speed. The surrogate's case rests on three situations: black-box models where only perturbation scans are possible, nonlinear time-domain behaviour (T3), and design search needing very many evaluations. Speed comparisons will be reported against both the white-box and the black-box solver paths, not only the favourable one.
- The reference solver stands in for a vendor EMT model. Agreement between surrogate and solver says nothing about agreement between solver and a real plant. That gap is covered only by Phase 2.
- A surrogate trained on one plant type is not expected to transfer to another. Phase 8 measures how badly.

## 10. Standards and practice referenced

To be read and cited properly in Phase 1; listed here as the frame the work sits in.

- IEEE Std 2800-2022, interconnection requirements for inverter-based resources.
- NERC reliability guidelines and event reports on inverter-based resource performance and sub-synchronous oscillation.
- ERCOT Nodal Protocols section on subsynchronous resonance study requirements for series-compensated corridors.
- IEEE PES technical report on wind energy systems sub-synchronous oscillations.
- IEEE SSR working group benchmark model papers (first and second benchmark models).

## 11. Starting references

Citations to be confirmed with DOIs in Phase 1.

- IEEE SSR Task Force, "First benchmark model for computer simulation of subsynchronous resonance," IEEE Trans. Power Apparatus and Systems, 1977.
- L. Fan, R. Kavasseri, Z. Miao, C. Zhu, "Modeling of DFIG-based wind farms for SSR analysis," IEEE Trans. Power Delivery, 2010.
- A. Ostadi, A. Yazdani, R. Varma, "Modeling and stability analysis of a DFIG-based wind-power generator interfaced with a series-compensated line," IEEE Trans. Power Delivery, 2009.
- J. Adams, C. Carter, S.-H. Huang, "ERCOT experience with sub-synchronous control interaction and proposed remediation," IEEE PES T&D, 2012.
- Y. Cheng et al., "Real-world subsynchronous oscillation events in power grids with high penetrations of inverter-based resources," IEEE Trans. Power Systems, 2023.
- L. Lu et al., "Learning nonlinear operators via DeepONet," Nature Machine Intelligence, 2021.
- Z. Li et al., "Fourier neural operator for parametric partial differential equations," ICLR, 2021.
- M. Raissi, P. Perdikaris, G. Karniadakis, "Physics-informed neural networks," J. Computational Physics, 2019.
- R. Chen et al., "Neural ordinary differential equations," NeurIPS, 2018.
- B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," NeurIPS, 2017.
- A. Angelopoulos, S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," 2021.

## 12. Risks

| Risk | Effect | Response |
|---|---|---|
| DFIG model does not reproduce published eigenvalues | Phase 2 stalls | System A proves the network and solver first, so the fault is isolated to the DFIG and its controls. Compare one subsystem at a time |
| No independent EMT tool models System B without major work | Validation rung 4 weakens | Second implementation in the abc frame, which is planned as the fallback and gives the discretization study anyway |
| No usable public measured waveforms | Validation rung 5 reduces to frequency and growth rate from published figures | Already assumed. Stated as a limitation, not hidden |
| Mode switching makes T1 targets discontinuous | Poor accuracy near switches | Mode-tracked targets; classification head for the active mode |
| 8 GB GPU memory limits T3 models | Smaller models or shorter windows | Mixed precision, windowed training, gradient checkpointing. Report the limit |
| Batched GPU integration disagrees with the CPU reference for stiff cases | T3 data wrong | Benchmark in Phase 3 against Radau; fall back to CPU generation |
| Scope is large for one person | Later phases slip | Minimum complete path defined in the schedule |
