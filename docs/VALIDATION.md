# Validation record

Status of each rung of the validation ladder in `docs/PLAN.md`, with the number that supports it and where it comes from. "Open" means not done. Tests named here fail if the number changes.

## Rung 1: hand calculations

| Check | System | Result | Source |
|---|---|---|---|
| Torsional frequencies against published values | A | within 0.005 Hz on five modes | step 01, `test_torsional_frequencies_match_published` |
| Slip estimate of decay rate against locked-shaft eigenvalue | A | within 2.6 % | step 01 |
| Self-excitation threshold in network resistance | A | 0.00613 pu by hand, 0.00606 pu from the model | step 01 |
| Damping-torque estimate of torsional growth | A | within 0.04 1/s for weak coupling; 86 % high at the mode 1 peak | step 01, A6 |
| Equivalent-circuit root against eigenvalue, speed held | B1 | equal to 3e-14 1/s | step 02 |
| Self-excitation boundary, hand against solver | B1 | within 0.9 points of compensation with the rotor free | step 02 |
| Energy from the waveform against `I² R_machine` | B1 | 1.4 % for a sustained oscillation; 2.1 % at complex frequency otherwise | step 02 |
| Controller gain as rotor resistance: every eigenvalue is a root of the circuit | B2, current loop | residual below 5e-11 pu | step 03 |
| Largest stable gain, hand against solver | B2, current loop | within 1.9 % at tuned `Ki / Kp`; fails at 100 times | step 03 |
| Energy budget of an oscillation | B2, current loop | closes to 7e-7 | step 03 |
| Hand boundary against full controls | B2, full | within 4 points with voltage feed-forward; up to 32 points without | step 04 |

## Rung 2: internal consistency

| Check | System | Result | Source |
|---|---|---|---|
| Steady state satisfies `dx/dt = 0` | A, B1, B2 | 3e-13, 4e-13, 2e-13 | steps 01, 02, 04 |
| Steady-state power balance | B1, B2 full | 1e-12, 1e-15 | steps 02, 04 |
| Autodiff Jacobian against finite differences | A, B1 | within 1e-5 | tests |
| Time-domain growth rate against eigenvalue | A | 6e-7 relative, ten cases | step 01 |
| | B1 | 1.1e-7 1/s, five cases | step 02 |
| | B2 current loop | 7e-8 1/s, four cases | step 03 |
| | B2 full | 7e-8 1/s, four cases | step 04 |
| Large disturbance: compensation step, growth against eigenvalue | B2 full | 3.479 against 3.474 1/s | step 04 |
| RK4 order of convergence | A | 4.00 | step 01 |
| CPU against GPU | A, B1, B2 | agree to 1e-11 or better | tests marked `gpu` |
| Reduced models recovered from fuller ones | B2 current loop → B1 | 1e-12 1/s | step 03 |
| | B2 full → B2 current loop | 0.0007 1/s | step 04 |

## Rung 3: published results

| Check | Status |
|---|---|
| First Benchmark Model torsional frequencies | Done (rung 1) |
| First Benchmark Model: benchmark compensation destabilises the second torsional mode | Reproduced: +0.358 1/s. Published growth rate not compared |
| First Benchmark Model generator and network data against the 1977 tables | **Open.** Entered from memory (A11) |
| Doubly-fed machine: direction of trends with compensation, rotor speed and current-loop gain | Reproduced (steps 03, 04) |
| Doubly-fed machine: eigenvalue tables and parameter sets | **Open.** Open copies of the reference paper were not retrievable. DOIs: 10.1109/TPWRD.2010.2050912, 10.1109/TEC.2011.2160995, 10.1109/TEC.2012.2211019 |
| Induction machine data against a source | **Open** (A12) |

## Rung 4: independent formulation

| Check | Result | Source |
|---|---|---|
| Impedance model (polynomial roots) against state-space eigenvalues, current-loop model, radial network | all four modes within 2.3e-8 on 27 cases | step 05, `test_impedance_roots_equal_state_space_eigenvalues` |
| Impedance model against full controls | **Open.** The impedance model has no phase-locked loop or outer loops |
| Second time-domain implementation in phase coordinates (NumPy, trapezoidal) against the D-Q model, current-loop model | growth rates within 0.00014 1/s on four cases; trajectories within 0.094 % of the largest excursion; order 2.00 | step 07, `test_growth_rate_from_phase_current_matches_the_eigenvalue` |
| Second time-domain implementation of the full controls, or an electromagnetic-transient tool | **Open** |
| Impedance model against the time-domain model on networks with plant impedance and a parallel path, added loops off | within 0.0005 1/s on four networks and three speeds | step 05, `test_controls_off_matches_the_impedance_model_on_the_meshed_network` |
| Meshed-network model reduces to the radial model | 1e-5 1/s | step 05 |
| Compensation step on the meshed network, growth against eigenvalue | 1.613 against 1.616 1/s | step 05 |
| Units online in the time-domain solver | **Open.** Rests on the impedance model alone |

## Rung 5: recorded events

**Open.** No digitised waveform has been compared. Public recordings come without plant parameters, so only frequency and growth rate could be compared.

## Assumption studies

| Study | Status |
|---|---|
| Rotor speed held against free | Done: at most 0.9 points of compensation (B1), +0.5 1/s (B2) |
| Network resistance, plant impedance, parallel path, units online | Done with the impedance model; plant impedance and parallel path confirmed in the time-domain model (step 05) |
| Voltage feed-forward on the grid-side converter | Done (step 04) |
| Time step; trapezoidal against RK4 | Done on System B2 (step 06): RK4 within 0.01 1/s up to 2 ms, trapezoidal up to 1 ms, in the rotating frame. Trapezoidal in phase coordinates: up to 0.2 ms (step 07). Order study on System A |
| Dropping stator and network transients | Done on the linearised model (step 06): the sub-synchronous mode disappears; all 28 unstable cases in 54 are reported stable; slow modes kept within 0.41 1/s in stable cases. Step 21, 38,511 cases: the slow mode below 2 Hz is kept within 0.006 1/s at the 99th percentile and 0.13 1/s at worst, with 7 wrong verdicts on it |
| Generic converter (System C): state matrix against admittance and impedance scans | Done (step 17): the Nyquist count from the scans agrees with the eigenvalues in all 60,124 cases; the scan determinant vanishes at the modes (a test). No comparison with a published case or a second time-domain implementation yet |
| Doubly-fed plant (System B2, full controls): state matrix against a 2 × 2 admittance scan of the plant and the impedance of the network | Done (step 18): the scan determinant vanishes at the modes (a test); the Nyquist count agrees with the eigenvalues in 33,579 of 33,580 cases, and in 4,892 of 4,931 placed on the boundary, the rest growing at less than 0.01 1/s |
| Growth rate from a rational fit to a 251-point scan against the eigenvalues, both plant types | Done (step 19): same sign in 98,635 of 98,635 cases; within 0.01 1/s in 99.9 % of cases in the range. A third route to the modes, independent of the state matrix and of the pointwise count |
| A plant model closed on the network by writing out the network equations, against the full state matrix | Done (step 20): every mode of the 25-state model reproduced to 1e-6 relative (a test), with and without a series capacitor. A least-squares fit of a clean 126-point scan, closed the same way, gives the growth rate in every band to better than 1e-6 1/s at the 99th percentile on 24,526 cases |
| Aggregation of unequal units into one machine | **Open.** Equal units scaled together is done (units online) |
| Saturation, limits, switching | **Open**, out of scope for now |

## What should not be quoted yet

- Absolute stability limits of the doubly-fed machine. They depend on network details and controller design choices that are not tied to a real plant.
- Any First Benchmark Model number that depends on generator reactances, until A11 is closed.
- Large-signal amplitudes. The models have no limits.
