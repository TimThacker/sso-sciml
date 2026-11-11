# Equations and conventions

Sections 1 to 11 cover Phase 1: the series-compensated line, the synchronous machine, the multi-mass shaft, and the two hand calculations (slip and damping torque) that the dynamic model is checked against. Sections 12 to 17 cover the induction generator, the energy accounting, the doubly-fed machine (current loop only, then full controls), the impedance formulation and the meshed network added in Phase 2. Code references are to `src/sso_sciml/`.

## 1. Per-unit system and time

- Base frequency `f0 = 60 Hz`, `wb = 2π f0` rad/s.
- Voltages, currents, fluxes, impedances and torque are in per unit on the generator rating.
- Time is in seconds. Because reactances are per-unit values at base frequency, every flux equation carries a factor `wb`: `dψ/dt = wb · (voltage terms)`.
- Speed is per unit. `dw_i = ω_i − 1` is the speed deviation of mass `i`.
- Angles are electrical radians.

## 2. Reference frames

Two rotating frames are used.

- **Synchronous frame (D, Q)**, rotating at `wb`. The infinite bus voltage lies on the D axis: `V_b = v_b + j0`. The series capacitor voltage is a state in this frame.
- **Rotor frame (d, q)**, fixed to the generator rotor. The q axis leads the d axis by 90°. `δ` is the angle of the q axis ahead of the D axis.

A complex quantity moves from the synchronous frame to the rotor frame by

```
(f_d + j f_q) = (f_D + j f_Q) · e^{−j(δ − π/2)}
```

which gives, for the voltage behind the line with `A = v_cD + v_b` and `B = v_cQ`:

```
v_d = A sin δ − B cos δ
v_q = A cos δ + B sin δ
```

and for the line current going the other way:

```
i_D =  i_d sin δ + i_q cos δ
i_Q = −i_d cos δ + i_q sin δ
```

A stator-frame oscillation at frequency `f_e` appears in either rotating frame at `f0 − f_e`. A sub-synchronous resonance at 39 Hz is a 21 Hz mode in the model.

## 3. Sign convention

Generator convention on the stator: positive current flows out of the machine. Air-gap torque `T_e` is positive when it decelerates the rotor.

## 4. Synchronous machine

Two rotor circuits per axis: field `fd` and one damper `1d` on the d axis, two dampers `1q`, `2q` on the q axis.

Flux linkages (d axis; the q axis has the same form with `X_aq`, `1q`, `2q`):

```
ψ_d  = −X_d  i_d + X_ad i_fd + X_ad i_1d
ψ_fd = −X_ad i_d + (X_ad + X_fd) i_fd + X_ad i_1d
ψ_1d = −X_ad i_d + X_ad i_fd + (X_ad + X_1d) i_1d
```

Voltage equations, stator transients kept:

```
(1/wb) dψ_d/dt  = v_d + R_a i_d + ω ψ_q
(1/wb) dψ_q/dt  = v_q + R_a i_q − ω ψ_d
(1/wb) dψ_fd/dt = v_fd − R_fd i_fd
(1/wb) dψ_1d/dt = −R_1d i_1d          (same for 1q, 2q)
```

Torque: `T_e = ψ_d i_q − ψ_q i_d`.

The `dψ_d/dt` and `dψ_q/dt` terms are the ones a positive-sequence phasor program sets to zero. Without them the network has no dynamics and the sub-synchronous mode does not exist.

### Standard parameters to circuit parameters

`machine.axis_from_standard` uses the classical relations:

```
X_ad = X_d − X_l
X_fd = X_ad (X_d' − X_l) / (X_ad − (X_d' − X_l))
X_1d = (X_d'' − X_l) X_ad X_fd / (X_ad X_fd − (X_d'' − X_l)(X_ad + X_fd))
R_fd = (X_ad + X_fd) / (wb T_d0')
R_1d = (X_1d + X_ad X_fd / (X_ad + X_fd)) / (wb T_d0'')
```

These reproduce `X_d`, `X_d'`, `X_d''` exactly and the time constants approximately (assumption A4).

### Operational reactance

With the rotor circuits shorted and a rotor-frame complex frequency `s` (per unit):

```
X_d(s) = X_d − s X_ad² · 1ᵀ (s L_rr + R_rr)⁻¹ 1
```

`X_d(0) = X_d`, `X_d(∞) = X_d''`. Implemented in `AxisCircuit.operational_reactance`.

## 5. Series-compensated line

Series `R_E`, `X_E`, `X_C` between the generator terminal and the infinite bus. `X_E` is the transformer, line and system reactance together.

The line inductance carries the stator current, so it is not an independent state. It is merged into the stator flux:

```
ψ_d' = ψ_d − X_E i_d        ψ_q' = ψ_q − X_E i_q
```

The stator equations keep their form with `ψ → ψ'`, `R_a → R_a + R_E`, `X_d → X_d + X_E`, and the terminal voltage replaced by the voltage behind the line, `V_c + V_b`. The torque expression is unchanged because the `X_E` terms cancel.

Capacitor, synchronous frame:

```
dv_cD/dt = wb (X_C i_D + v_cQ)
dv_cQ/dt = wb (X_C i_Q − v_cD)
```

A series R-L-C on its own has, in the rotating frame, two mode pairs at `f0 − f_er` and `f0 + f_er`, where `f_er = f0 √(X_C / X)`, both decaying at `R wb / (2X)`.

## 6. Shaft

Six masses in a chain: HP, IP, LPA, LPB, GEN, EXC.

```
dδ_i/dt       = wb · dw_i
2 H_i d(dw_i)/dt = T_i − D_i dw_i + K_{i−1}(δ_{i−1} − δ_i) − K_i(δ_i − δ_{i+1})
```

`T_i` is the turbine torque share on the four turbine masses, `−T_e` on the generator, zero on the exciter. `K` is in per-unit torque per electrical radian.

Natural frequencies come from the eigenvalues of `M^{-1/2} K M^{-1/2}` with `M = diag(2H_i / wb)` (`shaft.torsional_modes`).

## 7. State vector

Twenty states: `ψ_d', ψ_q', ψ_fd, ψ_1d, ψ_1q, ψ_2q, v_cD, v_cQ, δ_1..6, dw_1..6`. Defined in `physics/fbm.py`.

## 8. Steady state

For a given terminal power, power factor and voltage (`FBM.equilibrium`):

1. `I = conj(S / V_t)`.
2. `V_b = V_t − (R_E + j(X_E − X_C)) I`. Rotate everything so `V_b` is on the D axis.
3. `E_Q = V_t + (R_a + jX_q) I` locates the q axis: `δ = arg(E_Q)`.
4. Transform `I` and `V_t` to the rotor frame. `ψ_d = v_q + R_a i_q`, `ψ_q = −(v_d + R_a i_d)`, `i_fd = (ψ_d + X_d i_d) / X_ad`. Damper currents are zero.
5. `v_fd = R_fd i_fd`, `T_m = T_e`, `V_c = −jX_C I`.
6. Shaft angles from the torque each section carries.

## 9. The slip calculation (induction generator effect)

Implemented in `physics/slip.py`. It uses only the equivalent circuit.

A stator current at frequency `ω_e` (per unit, below 1) is seen from a rotor turning at `ω_r` at the frequency `ω_e − ω_r`, which is negative. The slip is

```
s = (ω_e − ω_r) / ω_e
```

In the rotor frame the stator voltage is `v = −(s_r + jω_r) X(s_r) i` with `s_r = j(ω_e − ω_r)`, so the stator-frame impedance of the machine is

```
Z_m(jω_e) = R_a + jω_e · X(j(ω_e − ω_r)),      X = (X_d(·) + X_q(·)) / 2
```

Because the rotor-frame frequency is negative, the imaginary part of `X` changes sign and the real part of `Z_m` is negative. That is the negative resistance.

Loop impedance: `Z(jω) = R_E + jωX_E − jX_C/ω + Z_m(jω)`.

- Resonance `ω_er`: `Im Z = 0`.
- Net resistance: `R_net = Re Z(jω_er)`. Self-excitation when `R_net < 0`.
- Hand growth rate: `σ = −wb · R_net / (d Im Z / dω)`. For a plain R-L-C this is `−R wb / (2X)`.
- Circuit root: Newton iteration on `Z(s) = 0` in the complex plane.

Averaging `X_d` and `X_q` drops the coupling between sub- and super-synchronous components (assumption A5).

## 10. The damping-torque calculation (torsional interaction)

Implemented in `physics/torsional.py`.

Impose a generator angle oscillation `Δδ(s)` with `Δω = sΔδ / wb` on the electrical subsystem and read the torque:

```
K_e(s) = ΔT_e / Δδ = c (sI − A_ee)⁻¹ (b_δ + (s/wb) b_ω)
```

`A_ee` is the 8×8 electrical block of the state matrix; `b_δ`, `b_ω` are its coupling columns; `c = ∂T_e/∂x_e`.

Electrical damping coefficient: `D_e(ω) = Im K_e(jω) · wb / ω`.

For torsional mode `n` with shape `φ` and modal inertia `H_n = Σ H_i φ_i²`, first-order perturbation gives

```
σ_n = −D_e(ω_n) φ_gen² / (4 H_n)
```

This holds when the electrical resonance and the torsional mode are weakly coupled (assumption A6).

## 11. Numerical methods

- Linearisation: `torch.func.jacrev` on the right-hand side, float64.
- Eigenvalues and participation factors: SciPy on the CPU. Participation of state `k` in mode `i` is `|w_ki v_ki|`, normalised per mode.
- Time integration: classical fourth-order Runge-Kutta, fixed step, batched. References: SciPy Radau and DOP853.
- Mode estimation from a signal: matrix pencil.

## 12. Induction generator (System B1)

Implemented in `physics/induction.py`. Single-cage rotor, round, so the whole model is written in the synchronous frame as complex quantities `f = f_D + j f_Q`. No rotor angle is needed.

Generator convention on the stator (current `i` positive out of the machine), rotor current `i_r` positive into the rotor. With `X_s = X_ls + X_m`, `X_r = X_lr + X_m`:

```
ψ_s = −X_s i + X_m i_r
ψ_r = −X_m i + X_r i_r
```

Voltage equations, rotor shorted, rotor speed `ω_r`:

```
(1/wb) dψ_s/dt = v_s + R_s i − j ψ_s
(1/wb) dψ_r/dt = −R_r i_r − j (1 − ω_r) ψ_r
```

The line is merged into the stator flux as in section 5: `ψ' = ψ_s − X_E i`, `R = R_s + R_E`, and the stator voltage becomes `v_b + v_c`:

```
(1/wb) dψ'/dt  = v_b + v_c + R i − j ψ'
(1/wb) dv_c/dt = X_C i − j v_c
```

Currents from fluxes, in closed form, with `X_ss = X_s + X_E` and `Δ = X_m² − X_ss X_r`:

```
i   = (X_r ψ' − X_m ψ_r) / Δ
i_r = (X_m ψ' − X_ss ψ_r) / Δ
```

Torque and motion (single mass, constant mechanical torque):

```
T_e = Im(conj(ψ') i) = ψ'_D i_Q − ψ'_Q i_D
2H dω_r/dt = T_m − T_e
```

State vector: `ψ'_D, ψ'_Q, ψ_rD, ψ_rQ, v_cD, v_cQ, ω_r`.

Terminal voltage, needed for the energy accounting, includes the inductive drop of the line:

```
v_s = v_b + v_c + R_E i + X_E ((1/wb) di/dt + j i)
```

with `di/dt` taken from the right-hand side through the flux map above.

### Steady state

At slip `s0 = 1 − ω_r` the loop is the usual equivalent circuit:

```
Z = R_E + R_s + j(X_E − X_C + X_ls) + jX_m ∥ (R_r/s0 + jX_lr)
i = −v_b / Z,      i_r = (v_b / Z) · jX_m / (R_r/s0 + j(X_m + X_lr))
```

Fluxes follow from the flux map, `v_c = −jX_C i`, `T_m = T_e`. For a given power into the infinite bus, `s0` is found on the stable side of pull-out (`InductionGenerator.equilibrium`).

### Slip calculation

Stator impedance at stator-frame complex frequency `s` (per unit):

```
Z_m(s) = R_s + s X_ls + (s X_m) ∥ (R_r / slip + s X_lr),      slip = (s − jω_r) / s
```

At `s = jω` with `ω < ω_r` the slip is real and negative, so the rotor branch carries negative resistance. The loop analysis of section 9 is then applied unchanged (`slip.analyse_loop`).

With `ω_r` held constant the electrical model is linear with complex coefficients, and its eigenvalues `λ` are exactly the roots of `Z(s) = 0` through `s = λ/wb + j`. The boundary `R_net = 0` is therefore exact for the electrical subsystem, where for the salient synchronous machine it was an approximation.

## 13. Energy accounting

Implemented in `physics/energy.py`.

Let `Δi` and `Δv` be the deviations of stator current and terminal voltage from the steady state. The power the machine delivers through the oscillation itself is

```
p_osc = Re(Δv · conj(Δi))
```

This is second order in the perturbation. The first-order cross terms with the steady state average out over a cycle and are left out by construction.

If the machine presents impedance `Z_m` to the oscillation then `Δv = −Z_m Δi` in generator convention, and

```
p_osc = −Re(Z_m) |Δi|²
```

so the resistance implied by a waveform is `R_eff = −∫ p_osc dt / ∫ |Δi|² dt`. Energy lost in the line over the same window is `R_E ∫ |Δi|² dt`. The oscillation is sustained when the two are equal.

For an oscillation that grows or decays at rate `σ`, `Z_m` must be evaluated at the complex frequency `σ/wb + jω`, not at `jω`. The difference is the rate of change of stored magnetic energy. It was 13 % of the resistance at a decay rate of 2.4 1/s (`docs/steps/02-induction-generator.md`).

## 14. Doubly-fed machine with rotor current control (System B2, first stage)

Implemented in `physics/dfig.py`. The machine and line equations are those of section 12, with a rotor voltage `v_r` from the converter:

```
(1/wb) dψ_r/dt = v_r − R_r i_r − j (1 − ω_r) ψ_r
```

The controller acts on rotor current in the synchronous frame:

```
v_r   = v_ff + Kp (i_ref − i_r) + Ki x
dx/dt = i_ref − i_r
```

`Kp` is in per unit, `Ki` in per unit per second, `x` in per unit seconds. `v_ff` is a constant feed-forward, used only to hold the steady state when `Ki = 0`.

At this stage the frame is ideal, `i_ref` is constant, there is no grid-side converter, and `ω_r` is a parameter. State vector: `ψ'_D, ψ'_Q, ψ_rD, ψ_rQ, v_cD, v_cQ, x_D, x_Q`. The model is linear in the state.

### Steady state

For stator power `P + jQ` into the infinite bus: `i = (P − jQ) / v_b`, `v_c = −jX_C i`, `ψ' = −j (v_b + v_c + R i)`, `i_r = (ψ' + X_ss i) / X_m`, `ψ_r` from the flux map, `v_r = R_r i_r + j (1 − ω_r) ψ_r`. Then `i_ref = i_r` and `x = v_r / Ki`.

### Tuning

Pole cancellation on the rotor transient reactance `σX_r = X_r − X_m² / X_s`, for a closed-loop bandwidth `α` in rad/s:

```
Kp = α σX_r / wb        Ki = α R_r
```

so `Ki / Kp = R_r wb / σX_r`, 10.9 1/s for the base machine.

### Slip calculation with the controller

For a perturbation, `Δv_r = −(Kp + Ki / S) Δi_r` with `S = wb (s − j)` the synchronous-frame frequency in 1/s and `s` the per-unit stator-frame frequency. The rotor equation becomes that of a shorted rotor with resistance

```
R_eff(s) = R_r + Kp + Ki / (wb (s − j))
```

and the machine impedance is that of section 12 with `R_r → R_eff` (`slip.dfig_impedance`). This is exact for the model above.

- `Kp` is added rotor resistance. At negative slip it adds negative resistance at the stator.
- At `s = jω` the `Ki` term is `−j Ki / (wb (1 − ω))`: imaginary, so after division by slip it is reactive. It shifts the resonance and does not add resistance to first order. It grows without bound as `ω → 1`, which creates extra zeros of loop reactance near synchronous frequency. The resonance search therefore takes the lowest crossing from capacitive to inductive.

The hand approximation "`Kp` as rotor resistance" drops the `Ki` term. Its error against the solver is 2 % at the tuned ratio and grows with `Ki / Kp` (`docs/steps/03-dfig-current-control.md`).

### Energy budget

Implemented in `energy.flows`. With `Δ` denoting deviations from the steady state and motor-convention stator current `−Δi`:

```
to network     = ∫ Re(Δv_s conj(Δi)) dt
from shaft     = ω_r ∫ Im(conj(Δψ') Δi) dt
from converter = ∫ Re(Δv_r conj(Δi_r)) dt
copper         = R_s ∫ |Δi|² dt + R_r ∫ |Δi_r|² dt
stored         = [ Re(−conj(Δi) Δψ_s + conj(Δi_r) Δψ_r) / (2 wb) ] end − start

from shaft + from converter = to network + copper + stored
```

Each term is second order, a product of two deviations. The balance follows from the voltage equations because the inductance matrix is real and symmetric.

## 15. Doubly-fed machine with full controls (System B2, second stage)

Implemented in `physics/dfig_full.py`. Complex quantities are in the synchronous frame unless marked with a superscript `c`, which means the phase-locked-loop frame: `f^c = f · e^{−jθ}`.

### Network with a grid-side converter

Three inductive branches meet at the terminal: the stator, the grid-side filter (`R_f`, `X_f`, current `i_g` from converter to terminal) and the line, which carries `i_L = i + i_g`. The terminal voltage is eliminated by using two loop fluxes as states:

```
λ1 = ψ_s − X_E i_L                (stator plus line)
λ2 = X_f i_g + X_E i_L            (grid-side filter plus line)

(1/wb) dλ1/dt = v_b + v_c + R_s i + R_E i_L − j λ1
(1/wb) dλ2/dt = v_g − v_b − v_c − R_f i_g − R_E i_L − j λ2
(1/wb) dv_c/dt = X_C i_L − j v_c
```

With `a = X_E / (X_f + X_E)`, the combination `λ1 + a λ2` plays the part of `ψ'` in section 12 with `X_E` replaced by `X_E X_f / (X_f + X_E)`. That gives `i` and `i_r` in closed form, then `i_g = (λ2 − X_E i) / (X_f + X_E)`.

Terminal voltage, from the grid-side branch:

```
v_t = v_g − R_f i_g − X_f ((1/wb) di_g/dt + j i_g)
```

The current derivatives come from the flux derivatives through the same linear map. Every controller acts on states, filtered quantities or currents, so the right-hand side has no algebraic loop.

### Phase-locked loop

```
dθ/dt     = Kp_pll · v_tq^c + Ki_pll · x_pll
dx_pll/dt = v_tq^c
```

`v_tq^c` is the q component of terminal voltage in the loop's own frame; it is zero when the frame is aligned. Gains from a natural frequency `ωn` and damping `ζ`: `Kp_pll = 2ζωn`, `Ki_pll = ωn²`.

### Rotor-side converter

```
P_ref      = K_opt ω_r³
dP_f/dt    = (P_t − P_f) / T_f                 (same for Q_f)
i_rd,ref   =   Kp_o (P_ref − P_f) + Ki_o x_P       dx_P/dt = P_ref − P_f
i_rq,ref   = −(Kp_o (Q_ref − Q_f) + Ki_o x_Q)      dx_Q/dt = Q_ref − Q_f
v_r^c      = Kp (i_r,ref − i_r^c) + Ki x_r         dx_r/dt = i_r,ref − i_r^c
```

`P_t + jQ_t = v_t · conj(i_L)`. With the terminal voltage on the d axis, d-axis rotor current sets active power and q-axis rotor current sets reactive power; the minus sign is because more reactive power needs less q-axis rotor current.

### Grid-side converter and DC link

```
i_gd,ref = −(Kp_dc (v_dc,ref − v_dc) + Ki_dc x_dc)      dx_dc/dt = v_dc,ref − v_dc
i_gq,ref = 0
v_g^c    = ff · v_f^c + Kp_g (i_g,ref − i_g^c) + Ki_g x_g
dv_f^c/dt = (v_t^c − v_f^c) / T_ff
C_dc v_dc dv_dc/dt = P_rsc − P_gsc
P_rsc = −Re(v_r conj(i_r))        P_gsc = Re(v_g conj(i_g))
```

`ff` is 1 with terminal-voltage feed-forward and 0 without. Converter voltages are taken as ideal: the DC voltage does not limit or scale them.

Without feed-forward, a perturbation sees `Δv_g^c = −(Kp_g + Ki_g/S) Δi_g^c`: the converter is a resistance `Kp_g` behind `X_f`, connected across the terminals. With fast feed-forward the terminal voltage perturbation is cancelled and the converter approaches a current source.

### Motion

```
2H dω_r/dt = P_m / ω_r − T_e,      T_e = Im(conj(ψ_s) i),   ψ_s = λ1 + X_E i_L
```

### Steady state

For a rotor speed `ω_0`: `P = K_opt ω_0³`. Solve `(v_b + (R_E + j(X_E − X_C)) i_L) conj(i_L) = P + jQ_ref` for `i_L`; then `v_t` and `θ = arg(v_t)`. The grid-side current is `i_g = i_gd e^{jθ}` with `i_gd` chosen so that `P_rsc = P_gsc`. The rest follows as in section 14, and `P_m = T_e ω_0`.

### State vector

`λ1 (2), ψ_r (2), v_c (2), λ2 (2), ω_r, θ, x_pll, x_r (2), P_f, Q_f, x_P, x_Q, x_g (2), v_dc, x_dc, v_f (2)`: 23 states.

## 16. Impedance formulation

Implemented in `physics/circuit.py`. It gives the modes of the current-loop model of section 14 without a state vector.

Every element is a ratio of polynomials in the per-unit stator-frame frequency `s`:

```
series branch:   Z = R + sX + X_C / s
parallel:        Z_a ∥ Z_b = Z_a Z_b / (Z_a + Z_b)
machine:         Z_m(s) = R_s + sX_ls + (sX_m) ∥ (R_eff(s) · s / (s − jω_r) + sX_lr)
                 R_eff(s) = R_r + Kp + (Ki / wb) / (s − j)
```

The modes are the zeros of the total impedance seen around the loop:

```
Z_m(s) / n + Z_net(s) = 0,        λ = wb (s − j)
```

`n` is the fraction of units online: `n` identical machines in parallel on a fixed base have impedance `Z_m / n`.

Networks used:

```
radial:                Z_net = R_E + sX_E + X_C / s
with plant impedance:  Z_net = (R_E + sX_E + X_C / s) + (R_plant + sX_plant)
with a parallel path:  Z_net = (R_E + sX_E + X_C / s) ∥ (ρR_E + sρX_E)
```

Each root `s` gives one eigenvalue of the synchronous-frame model; the state-space model has that eigenvalue and its conjugate.

Building the total impedance by cross-multiplication leaves common factors in numerator and denominator (at `s = 0`, `s = j` and `s = jω_r`). Each denominator root removes the numerator root nearest to it when the two are within 1e-5. A true mode can sit near a pole when the gain is low, so the test is on distance between roots and not on the size of the denominator.

## 17. Full-control model on a meshed network

Implemented in `physics/dfig_meshed.py`. Controls and machine as in section 15. Network:

```
terminals — (R_t, X_t) — plant bus —+— compensated line (R_E, X_E, X_C) —+— infinite bus
                                    +— parallel path (R_p, X_p) ————————+
```

`i_L = i + i_g` flows through the plant impedance. `i_p` flows in the parallel path and `i_c = i_L − i_p` in the compensated line. The plant bus voltage is eliminated with a third loop flux:

```
λ1 = ψ_s − X_t i_L − X_E i_c
λ2 = X_f i_g + X_t i_L + X_E i_c
μ  = X_p i_p − X_E i_c

(1/wb) dλ1/dt = v_b + v_c + R_s i + R_t i_L + R_E i_c − j λ1
(1/wb) dλ2/dt = v_g − v_b − v_c − R_f i_g − R_t i_L − R_E i_c − j λ2
(1/wb) dμ/dt  = v_c + R_E i_c − R_p i_p − j μ
(1/wb) dv_c/dt = X_C i_c − j v_c
```

Fluxes and currents are related by

```
[λ1]   [−(X_s+X_t+X_E)   −(X_t+X_E)    X_m    X_E     ] [i  ]
[λ2] = [  X_t+X_E         X_f+X_t+X_E   0     −X_E     ] [i_g]
[ψ_r]  [ −X_m             0             X_r    0       ] [i_r]
[μ ]   [ −X_E            −X_E           0      X_p+X_E ] [i_p]
```

solved numerically at each evaluation. Stator flux for the torque: `ψ_s = λ1 + X_t i_L + X_E i_c`.

Steady state: both branches end at the infinite bus, so the network seen from the terminals is `v_b` behind `Z = R_t + jX_t + Z_c Z_p / (Z_c + Z_p)` with `Z_c = R_E + j(X_E − X_C)` and `Z_p = R_p + jX_p`. Then `i_c = i_L Z_p / (Z_c + Z_p)`, and the rest follows section 15.

State vector: the 23 states of section 15 plus `μ_D, μ_Q`.
