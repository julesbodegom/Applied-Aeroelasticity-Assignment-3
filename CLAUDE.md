# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coursework for **Applied Aeroelasticity (AE4ASM522) Assignment 3** — flexible-aircraft
flight dynamics. Python supporting code that builds linearised state-space models of a
glider (flexible cantilever-beam wings, rigid tail), then analyses them via eigenvalues
and time-domain simulation. Read `CONTEXT.md` before touching the model — it is the
authoritative glossary for the overloaded symbols (`w`, `phi`, `theta` mean different
things as nodal DOFs vs. rigid-body states) and the sign/mirror conventions.

The assignment PDF and folder names say "Assignment 2"; this is actually Assignment 3.

## Running

All code is plain scripts — there is no build, package install, or test suite.
Dependencies: `numpy`, `scipy`, `matplotlib`. No `requirements.txt`.

**Critical: scripts must run with the working directory set to the source folder.**
Imports are flat (`from Get_Glider_Properties import ...`) and data files are loaded by
bare relative path (`scipy.io.loadmat('aero_coeff.mat')`, `'trim_V35.mat'`). Running from
the repo root will fail on both imports and file loads.

```powershell
cd "AE4ASM522_Assignment2_Supporting_Code_Python\AE4ASM522_Assignment2_Supporting_Code_Python"
python main_Typical_2D_Section_Dynamics.py   # 2D typical section (simplest)
python main_Clamped_Wing_Dynamics.py         # clamped half-wing FEM beam
python main_Aircraft_Flight_Dynamics.py      # full free-flying aircraft
```

Each `main_*` script ends in `plt.show()` (blocking). There are no command-line args;
configuration is edited inline in the script.

## Entry points (complexity ladder)

The three `main_*` scripts are a deliberate progression — same aeroelastic assembly
pattern at increasing scale:

1. **`main_Typical_2D_Section_Dynamics.py`** — single 2D section, 3 structural DOFs
   (heave, pitch, flap). Self-contained; assembles the state-space inline rather than
   calling the `Get_State_Space_*` helpers. Good place to understand the quasi-steady
   vs. unsteady split (the `/0.5` factor on circulatory terms, the lag states).
2. **`main_Clamped_Wing_Dynamics.py`** — one FEM half-wing beam (`Nn=8` nodes), clamped
   at the root. Exercises both `Get_State_Space_Elastic_Wing` and `Get_State_Space_Rigid_Wing`.
3. **`main_Aircraft_Flight_Dynamics.py`** — full free-flying aircraft: couples left+right
   wings to the 12 rigid-body DOFs. Builds the two configurations the assignment requires.

## Architecture

The full-aircraft model is assembled by `Get_EQM_Coupled_Flight_Dynamics`, which blends
three independently-built pieces into one `A_Fdyn`:

- **Rigid-body EQM** (`Get_EQM_Rigid_Dof` + `Get_Aero_Jacobian_Tail`): the 12-state
  `M_rr \dot x = A_rr x + B_rr [delta_a, delta_e, delta_r]` plus the tail aero Jacobian.
- **Per-wing aeroelastic state-space** (`Get_State_Space_Elastic_Wing` or
  `Get_State_Space_Rigid_Wing`, fed by `Get_FEM_Beam_Structure` + `Get_Beam_Strip_Aerodynamics`):
  one wing's `\dot x_w = A_ww x_w + B_w_{r,g,f} u`, with outputs `y = [lift, bending, torsion]`
  integrated to the wing **root** via the `H_root` selection matrix. The same wing block is
  reused for both sides; only the transfer matrices differ.
- **Coupling matrices** (`Get_Coupling_Matrices`): the `R_AOA_{rw,lw}` (rigid motion →
  local strip AOA) and `R_FM_{rw,lw}` (wing root force/moment → rigid-body forcing)
  transfer matrices. The left/right asymmetry (roll-moment sign flip, see CONTEXT.md
  "Mirror rule") lives here, not in the wing block.

The assembled `A_Fdyn` is block `[[rigid, rw_out, lw_out], [rw_in, A_ww, 0], [lw_in, 0, A_ww]]`,
with input matrices `B_Fdyn_delta` (control surfaces), `B_Fdyn_flap` (per-strip flap moment,
14 inputs), `B_Fdyn_gust` (per-strip gust AOA, 14 inputs).

### The two options switches

`option['stru']` and `option['aero']` select the configuration (`0`/`1` — note they are
**inverted** between the two flags, easy to get wrong):

| `option` | meaning |
|----------|---------|
| `stru = 1` | **rigid** wing beam (only flap `beta` is a structural DOF) |
| `stru = 0` | **elastic** FEM wing beam (4 DOFs/node) |
| `aero = 1` | **quasi-steady** aerodynamics |
| `aero = 0` | **unsteady** aerodynamics (adds lag states) |

The assignment builds exactly two of the four combinations:
- **`Rigid_AC_Quasi_Aero`** = `stru=1, aero=1` → **40 states**.
- **`Flexible_AC_Un_Aero`** = `stru=0, aero=0` → **180 states**.

### State & input ordering

Authoritative ordering lives in `CONTEXT.md` and the header comments of `initialization.py`.
In short: 12 rigid states first, then right-wing then left-wing aeroelastic blocks. The
quasi-steady/unsteady distinction is implemented as the `Aero['Ca_cir']/0.5` and
`Ka_cir/0.5` scaling plus the presence/absence of lag-state rows — compare the two
branches of `Get_State_Space_Elastic_Wing` to see it.

### Data files (must be in the working directory)

- `aero_coeff.mat` — wing/tail lift-curve slopes and control derivatives (loaded in
  `Get_Glider_Properties`).
- `trim_V35.mat` — trim `alpha_trim`, `delta_e_trim`, `T_trim` at the reference V = 35 m/s.

### Conventions you must not get wrong

- **Units**: all states/angles are in **radians**; gust magnitude is given in **degrees**
  and flap input in **Nm** — convert with `d2r` before feeding inputs (the assignment
  explicitly warns about this).
- **Flap selection** `H_flap_selection` encodes "right tip down, left tip up" = positive
  aileron / roll-left. **Gust** is a uniform "1−cos" vertical gust (5°, 0.2 Hz).
- See `CONTEXT.md` for the full sign/mirror rules — overloaded `w/phi/theta`, the left↔right
  roll-moment flip, flap-down-is-positive-lift, etc.

## What the assignment actually asks (the deliverables)

The brief is `Assignment 2  Flexible Aircraft Flight Dynamics.pdf` — **mislabelled; this is
Assignment 3**. Most of the code is given and complete. The graded work is: fill in a few
marked lines, then produce eigenvalue plots and time-domain simulations with written
discussion. The three parts:

**Part A — Clamped wing dynamics (no coding).** Run `main_Clamped_Wing_Dynamics.py`. Plot
clamped-beam eigenvalues for rigid vs. flexible beam, each with quasi-steady vs. unsteady
aero, and discuss.

**Part B — Flight dynamic models. COMPLETE.**

- [`Get_Aero_Jacobian_Tail.py`](AE4ASM522_Assignment2_Supporting_Code_Python/AE4ASM522_Assignment2_Supporting_Code_Python/Get_Aero_Jacobian_Tail.py)
  — all six `tbc` derivatives are implemented: `Y_p_T = Y_v_T*(-r_cg_vt[2])`,
  `Y_r_T = Y_v_T*r_cg_vt[0]`, `Y_dr_T = -0.5*rho*V²*S_vt*Cl_dr`,
  `Z_q_T = Z_w_T*(-r_cg_ht[0])`, `Z_de_T = -0.5*rho*V²*S_ht*Cl_de`,
  `M_T = r_cg_ht[2]*X_T_ht + r_cg_vt[2]*X_T_vt - r_cg_ht[0]*Z_T`.
- [`Get_Coupling_Matrices.py`](AE4ASM522_Assignment2_Supporting_Code_Python/AE4ASM522_Assignment2_Supporting_Code_Python/Get_Coupling_Matrices.py)
  — left-wing block implemented: `R_F_lw` identical to `R_F_rw`; `R_M_lw` mirrors
  `R_M_rw` with `R_M_lw[0,1] = +1` (vs `-1`) to flip the bending-to-roll sign.

`main_Aircraft_Flight_Dynamics.py` now builds both configs, identifies the five rigid-body
modes by **eigenvector participation** (`p_ij = |v_ij| / Σ_k|v_kj|`, summed over the body
velocity/rate states `[u,v,w,p,q,r]`; near-zero eigenvalues are the position/heading
integrators), and saves an annotated two-panel complex-plane figure (full spectrum +
flight-dynamics zoom) to `report/figures/partB_aircraft_eigenvalues.pdf`. Mode classification:
short-period ↔ `w,q`; phugoid ↔ `u`; Dutch roll ↔ `v,r`; roll ↔ fast real `p`; spiral ↔
slow real bank/heading. Flexible wing-bending modes are picked out by their nodal-DOF
participation and left↔right symmetry.

Key results (V = 35 m/s): the **spiral is mildly unstable** (+0.018) in *both* aircraft, so
flexibility doesn't change the stability verdict. Flexibility **dissolves the short-period**
(its heave/pitch content is absorbed into the symmetric wing-bending modes near −1.86 and
−2.45 ± 4.68j — no flexible mode keeps rigid pitch content) and **collapses roll-subsidence
damping** (−49 → ≈ −8, smeared across several aeroelastic poles), while Dutch roll, the
(overdamped) phugoid and the spiral are nearly unchanged. Written up in
`report/partB_implementation.tex` (§ Eigenvalue analysis); the LaTeX report root is
`report/main.tex`.

**Part C — Time-domain simulation (coding required).** `initialization.py` is "fully coded
for you" per the brief (it builds the sorted input matrices `Rigid_B_Fdyn_sorted` /
`Flex_B_Fdyn_sorted`); you add "a few lines" to integrate and plot. Suggested solver
settings from the brief: **ode4 / RK4, 20 kHz sampling, 8 s**. Six cases:

1. zero input (sanity check — all states should stay zero);
2. elevator pulse (5° from t=1s to t=3s);
3. rudder pulse (5°, same timing);
4. asymmetric flap pulse (right tip flaps +10 Nm, left tip −10 Nm, t=1–3s — the
   `H_flap_selection` aileron case);
5. "1−cos" gust, 5°, 0.2 Hz, one period then zero, run long enough to see post-gust response;
6. sharper gust, 5°, 0.5 Hz.

For each case: plot rigid-body states `[u v w p q r]` for both aircraft; for the flexible
aircraft also plot nodal bending displacements and torsion angles (left/right differ under
asymmetric input).

### Watch out

- **NumPy ≥2.0 scalar assignment**: the `.mat` scalars (`alpha_trim`, `Clalpha_W`,
  `Cl_delta_e`, …) load as `(1,1)` arrays. Assigning them into a scalar matrix slot
  (e.g. `R_F_rw[0,0] = ... * par['Clalpha_W'] * par['alpha_trim']` in `Get_Coupling_Matrices`)
  raises `ValueError: setting an array element with a sequence` on NumPy ≥2.0 — the given
  code was written for older NumPy. **Fixed at source** in `Get_Glider_Properties` by
  wrapping the 5 aero coefficients and 3 trim values in `float(np.squeeze(...))`. Don't
  reintroduce raw `loadmat` scalars; the tail Jacobian's `np.mean(...)` wrappers also rely
  on these being plain floats.
- `initialization.py` indexes states/inputs with **MATLAB 1-based** ranges
  (`np.arange(1,13)`, `index_right_flap = np.arange(20,27)`, etc.) and uses **attribute**
  access (`Rigid_AC_Quasi_Aero.A_Fdyn`), but `Get_EQM_Coupled_Flight_Dynamics` returns a
  **dict** (`['A_Fdyn']`). Reconcile both (0-based indices, dict access) when wiring up the
  simulation, or wrap the dicts in an object.
- The model-saving line in `main_Aircraft_Flight_Dynamics.py` (`np.savez(...)`) is commented
  out; `initialization.py`'s commented `loadmat('AC_Fdyn_Sys_SS.mat')` expects a saved model
  — decide whether to save-then-load or build the models in-process.

### Reference PDFs (repo root)

- `Assignment 2  Flexible Aircraft Flight Dynamics.pdf` — the brief (this is Assignment 3).
- `AE4ASM522_AAA_Guideline_for_the_aerodynamic_Jacobian_of_the_tail.pdf` — for the
  `Get_Aero_Jacobian_Tail` `tbc` terms.
- `AE4ASM522_AAA_Guideline_for_the_Assembling_Procedure.pdf` — for the coupling matrices.
- `Waszak, Schmidt - 1988 - Flight Dynamics of Aeroelastic Vehicles.pdf` — background theory.

