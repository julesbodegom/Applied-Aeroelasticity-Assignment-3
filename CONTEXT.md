# Applied Aeroelasticity Assignment 3 — Flexible Aircraft Flight Dynamics

A glider model whose wings are flexible cantilever beams and whose tail is rigid.
The work assembles linearised free-flying flight dynamic equations and analyses
them through eigenvalue and time-domain simulation. (The assignment PDF is mislabelled
"Assignment 2"; this is Assignment 3.)

## Language

### Nodal DOFs

Each FEM beam node carries four structural DOFs, ordered `(w, phi, theta, beta)`
in code (`EFT_beam` index 0..3). Confirmed against the Euler–Bernoulli consistent
mass block in `Get_FEM_Beam_Structure`.

**w**:
Transverse (out-of-plane) bending **displacement** of the node. The assignment text
loosely calls this "heave". The report's "nodal bending displacements" = `w`.
_Avoid_: heave (reserve that for the rigid-body vertical velocity).

**phi**:
Bending **slope** `dw/dx` (the rotation paired with `w` in the beam element).
The assignment text calls this "out-of-plane bending".

**theta**:
Torsion (twist) angle of the section. The report's "torsion angles" = `theta`.

**beta**:
Flap deflection. The *only* structural DOF retained in the rigid-beam model.

### Rigid-body states

The 12 rigid-body states are ordered (per `Get_EQM_Rigid_Dof`):

```
[Px, Py, Pz | u, v, w | phi, theta, psi | p, q, r]
```

i.e. positions, then body-axis velocities, then Euler angles, then body rates.
The report plots the six-element subset `[u, v, w, p, q, r]` (states 4–6 and 10–12).

**Beware three overloaded symbols** — same letter, different quantity and frame
than the nodal DOFs above:

- `w` (rigid) = body vertical velocity; `w` (nodal) = bending displacement.
- `phi` (rigid) = roll Euler angle; `phi` (nodal) = bending slope.
- `theta` (rigid) = pitch Euler angle; `theta` (nodal) = torsion twist.

When ambiguous in prose, qualify as `w_body`/`phi_roll`/`theta_pitch` for the
rigid-body states.

### Wings, strips & sides

**Strip**:
A spanwise slice of wing whose 2D (sectional) aerodynamics are computed by
strip theory and attached at a node. One strip per node.

**Node / Element**:
FEM discretisation of the half-wing beam. `Nn = 8` nodes, `Ne = 7` elements;
each node holds the four DOFs `(w, phi, theta, beta)`.

**Right wing / Left wing**:
The two half-wings of the free-flying aircraft. They are geometric mirror images
about the body x–z plane: identical chordwise (x) and dihedral-drop (z) arms,
opposite spanwise (y) arm.
_Mirror rule for aero→body transfer_: **force** directions are identical on both
sides (lift up, induced drag aft — body-frame directions are side-independent);
**moment** transfer flips the **roll** contribution sign and keeps the **pitch**
contribution (pitch uses the side-independent x-arm). The spanwise roll arm is
handled separately through each wing's own cg coordinates.

### Trim condition

The steady reference flight state (here V = 35 m/s) about which the equations are
linearised. All state-space states are *perturbations* around trim; trim angle of
attack, elevator, and thrust come from `trim_V35.mat`.

### Inputs, gust & units

**Flap input sign**:
A positive flap hinge moment deflects the flap trailing-edge **down** → more lift on
that strip. The case-4 aileron-like input (right tip +, left tip −) is therefore a
**positive-aileron / roll-left** command: right wing gains lift, left loses it, the
left wing drops (sign of `p` per the body convention where `p>0` = right wing down).

**Gust**:
A "1−cos" vertical gust giving a prescribed local angle-of-attack perturbation,
uniform along the span. One period only, then zero, with the run continued to see
the post-gust response. Magnitude 5°; frequency 0.2 Hz (sharper case: 0.5 Hz).

**Units**:
All model states and angles are in **radians**; gust magnitude is specified in
**degrees** and flap input in **Nm**. Convert deg→rad before feeding inputs into
the state-space (the assignment explicitly warns about this).

### Model axes

Two orthogonal modelling choices combine to define a configuration:

**Structural model**:
Whether the wing is treated as a single **rigid beam** (only the flap input `beta`
is a structural DOF) or a **flexible** FEM beam (each node carries four DOFs).
_Avoid_: stiff/elastic used loosely.

**Aerodynamic model**:
Whether strip-theory loads are **quasi-steady** or **unsteady** (the latter adds
aerodynamic lag states).
_Avoid_: steady, dynamic.

### Configurations

The assignment fixes exactly two full-aircraft configurations; the other two
axis combinations are never built for the free-flying aircraft.

**Rigid aircraft**:
Rigid beam + quasi-steady aerodynamics. Code object: `Rigid_AC_Quasi_Aero`.
40 states.

**Flexible aircraft**:
Flexible FEM beam + unsteady aerodynamics. Code object: `Flexible_AC_Un_Aero`.
180 states.

### Rigid-body (flight-dynamic) modes

The classical aircraft modes that live in the low-frequency, near-origin region of
the eigenvalue spectrum — distinct from the high-frequency structural and aerodynamic-lag
poles. They are **identified by participation factor** (see below): which body-axis
velocity/rate states (`[u, v, w, p, q, r]`, *not* the position/angle integrators)
dominate a given eigenvalue.

- **Short-period**: well-damped oscillation; dominant `w` and `q` (longitudinal).
- **Phugoid**: low-frequency longitudinal mode in `u` (and pitch). May appear as a
  complex pair or, for this glider, **overdamped into two real roots**.
- **Dutch roll**: lateral oscillation; dominant sideslip `v` with `r`, `p`.
- **Roll subsidence**: fast real (heavily negative) mode; dominant `p`.
- **Spiral**: slow real mode near the origin (heading/yaw); for this model it is
  **mildly unstable** (positive real part) — the headline stability finding for Part B.

The near-zero eigenvalues are the kinematic integrators (positions `Px,Py,Pz`, heading
`psi`), not dynamic modes.

**Participation factor**:
The scale-invariant measure of how strongly state `k` participates in mode `i`:
`p_ki = (left eigenvector)_ki · (right eigenvector)_ki`, with the biorthogonal
normalisation `Σ_k p_ki = 1` per mode (and `Σ_i p_ki = 1` per state). The canonical
quantity for "which state dominates this mode", because it is invariant under any
diagonal state rescaling — unlike raw eigenvector magnitude, which mixes incommensurate
units (m, m/s, rad/s, rad).
_Avoid_: "mode-shape magnitude", "right-eigenvector magnitude" — that normalised
right-eigenvector quantity is unit-dependent and is the deprecated identification metric.

**Inverse participation ratio (IPR)**:
For a signature state `s`, `IPR(s) = 1 / Σ_i p̂_{s,i}²` over the normalised participations
across all modes — an "effective number of modes" state `s` lives in. `≈1` = the state's
motion is concentrated on a single eigenvalue; large = the signature is **smeared** across
many poles (e.g. roll-rate `p` in the flexible aircraft). Quantifies "a mode dissolves /
collapses" as a number rather than an assertion.

**Isolated rigid-body block**:
The bare 12-state flight-dynamics system (`Get_EQM_Rigid_Dof` + tail aero Jacobian, **no
wing aerodynamic feedback**). Its five modes are textbook-unambiguous, so it serves as the
independent baseline against which the coupled-aircraft mode labels are verified. Because it
omits wing aero, modes whose damping is wing-dominated (roll subsidence `L_p`, short-period
heave `Z_w`) deliberately differ from the coupled system — that gap *measures* the wing's
contribution.
_Avoid_: confusing this with the coupled "Rigid aircraft" config (which has rigid-beam
wings *with* their aero).
