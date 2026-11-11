# Assumption ledger

One row per modelling assumption: what it affects, how it was tested, and what the test showed. "Not tested" is a valid entry and means the risk is open.

| ID | Assumption | Affects | Test | Result |
|---|---|---|---|---|
| A1 | Balanced three-phase conditions; dq model with no zero sequence | Everything. Unbalanced faults cannot be studied | Not tested. Needs the abc-frame model planned for Phase 2 | Open |
| A2 | Stator and network flux transients are kept | Existence of the sub-synchronous mode | `test_series_rlc_modes_appear_at_f0_minus_and_plus_fer`: the network block gives modes at `f0 ± f_er` with decay `R wb / 2X` | Holds to 0.02 Hz and 0.2 % |
| A3 | Line is a lumped series R-L-C; no shunt capacitance, no frequency dependence of R | Resonant frequency and network damping | Not tested. A distributed-line comparison belongs in Phase 2 | Open |
| A4 | Machine circuit parameters from the classical conversion of standard parameters | Rotor resistances, and through them the negative resistance and all damping | Reactances checked at the limits `X(0)` and `X(∞)`. Time constants not compared with the exact conversion | Reactances exact. Time-constant error not quantified |
| A5 | Slip calculation averages the d and q operational reactances | Hand estimate of resonance and growth rate | Compared with the locked-shaft eigenvalue of the full electrical model over `X_C` = 0.02 to 0.62 pu | Growth rate within 2.6 %, frequency within 0.09 Hz |
| A6 | Damping-torque estimate is first-order in the coupling between electrical and torsional modes | Torsional growth-rate estimate | Compared with full eigenvalues over the same sweep | Within 0.04 1/s where growth is below 0.2 1/s. Overestimates the mode 1 peak by 86 % (7.34 against 3.95 1/s). Peak location agrees |
| A7 | No mechanical damping (`D_i = 0`) | Torsional growth rates are worst-case; small positive values appear at every compensation level | By construction. Real units have small positive modal damping | Stated, not tested. Stability margins from this model are pessimistic |
| A8 | Constant field voltage and constant mechanical torque (no AVR, no governor) | Low-frequency swing mode; negligible above 10 Hz | Not tested | Open; expected small for torsional frequencies |
| A9 | No magnetic saturation | Operating-point reactances | Not tested | Open |
| A10 | Fixed-step RK4 at 0.1 to 0.2 ms | Time-domain accuracy | Convergence study against DOP853 at tight tolerance; observed order 4.00 | Error 1.5e-8 at 0.125 ms over 0.5 s |
| A11 | Benchmark data entered from memory of secondary sources | Every number for System A | Shaft data reproduce the five published torsional frequencies to 0.005 Hz, which pins inertias and stiffnesses jointly. Network and generator data have no such check | Shaft: confirmed indirectly. Network resistance split and generator data: to be checked against the 1977 paper |
