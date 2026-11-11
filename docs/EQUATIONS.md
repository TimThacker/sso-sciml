# Equations and conventions

Covers what Phase 1 implements: the series-compensated line, the synchronous machine, the multi-mass shaft, and the two hand calculations (slip and damping torque) that the dynamic model is checked against. Code references are to `src/sso_sciml/`.

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
