import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import scipy
from Get_Glider_Properties import Get_Glider_Properties
from Get_Coupling_Matrices import Get_Coupling_Matrices
from Get_Aero_Jacobian_Tail import Get_Aero_Jacobian_Tail
from Get_EQM_Coupled_Flight_Dynamics import Get_EQM_Coupled_Flight_Dynamics
from Get_EQM_Rigid_Dof import Get_EQM_Rigid_Dof
from skew import skew

# Flow parameters
flow={}
flow['rho'] = 1.225
flow['Vel'] = 35

# Glider Overall Properties
par = Get_Glider_Properties()

# Aerodynamic influences from the tail
Jco_Tail = Get_Aero_Jacobian_Tail(par, flow)

# Equations of motion for rigid-body degrees of freedom
Rigid_Dof = Get_EQM_Rigid_Dof(par, flow, Jco_Tail)

# Coupling matrices between rigid-body and aeroelastic Dof
Coup = Get_Coupling_Matrices(par, flow)

# Options
option={}
option['aero'] = 1 # 1-Quasi-steady aerodynamics  0-unsteady aerodynamics
option['stru'] = 1 # Rigid wing  0-elastic wing

# Fully coupled flight dynamic equations
Fdyn_Sys_SS = Get_EQM_Coupled_Flight_Dynamics(Rigid_Dof, Coup, flow, par, option)

# Model analysis
Rigid_AC_Quasi_Aero = Fdyn_Sys_SS

eig_rigid = np.linalg.eigvals(Rigid_AC_Quasi_Aero['A_Fdyn'])

option={}
option['aero'] = 0 # 1-Quasi-steady aerodynamics  0-unsteady aerodynamics
option['stru'] = 0 # Rigid wing  0-elastic wing
Flexible_AC_Un_Aero = Get_EQM_Coupled_Flight_Dynamics(Rigid_Dof, Coup, flow, par, option)

eig_flex = np.linalg.eigvals(Flexible_AC_Un_Aero['A_Fdyn'])


# ---------------------------------------------------------------------------
# Rigid-body (flight-dynamic) mode identification, by participation factor
# ---------------------------------------------------------------------------
# Modes are identified from scale-invariant modal participation factors
#   p_ki = (left eigvec)_ki * (right eigvec)_ki   (normalised so sum_k p_ki = 1),
# computed via scipy.linalg.eig(left=True). Unlike the normalised right-
# eigenvector magnitude used previously, participation factors are invariant
# under diagonal state rescaling, so "which state dominates a mode" is not an
# artefact of the incommensurate units (m, m/s, rad, rad/s, dimensionless lag)
# in the 180-state vector. See CONTEXT.md "Rigid-body (flight-dynamic) modes"
# and docs/adr/0001-mode-identification-by-participation-factors.md.
#
# Body-axis velocity/rate states in the 12-state rigid-body block (0-based):
#   [Px Py Pz | u v w | phi theta psi | p q r]
import scipy.linalg as sla

U, VY, WZ, PHI, THE, PSI, P, Q, R = 3, 4, 5, 6, 7, 8, 9, 10, 11
INTEG = [0, 1, 2, PSI]                         # kinematic integrators
VELRATE = [U, VY, WZ, P, Q, R]                 # dynamic discriminants


def participation_factors(A):
    """Scale-invariant participation factors. Returns (eigvals, Pmag, cond),
    where Pmag[k,i] = |p_ki| column-normalised (sum_k Pmag = 1 per mode) and
    cond = |vl_i^H vr_i| (small => near-defective; participation unreliable for
    that mode, e.g. the exactly-degenerate poles from the identical L/R wings)."""
    w, vl, vr = sla.eig(A, left=True, right=True)
    bi = np.sum(vl.conj() * vr, axis=0)        # biorthogonal normalisation
    # The per-mode scalar bi cancels under the column normalisation below, so we
    # never divide by it (it is ~1e-295 for the degenerate L/R-wing poles and
    # would overflow). |bi| is retained only as the conditioning measure.
    num = np.abs(vl.conj() * vr)
    Pmag = num / (num.sum(axis=0, keepdims=True) + 1e-300)
    return w, Pmag, np.abs(bi)


def inverse_participation_ratio(p_over_modes):
    """Effective number of modes a state's participation spreads over.
    ~1 = concentrated on one pole; large = smeared across many."""
    x = p_over_modes / (p_over_modes.sum() + 1e-30)
    return 1.0 / np.sum(x ** 2)


def identify_rigid_body_modes(A):
    """Identify the five classical modes from participation factors. Returns
    {name: (eigenvalue, info)}. Threshold-free: integrators are excluded by
    their own participation signature, each classical mode is the argmax-
    participation eigenvalue in its signature, split only by real vs complex.
    The roll mode is the global max-p eigenvalue -- it need NOT be real once the
    wing is flexible -- and carries its spread (IPR_p) and conditioning."""
    w, Pm, cond = participation_factors(A)
    cand = Pm[INTEG, :].sum(axis=0) <= 0.5     # not an integrator
    osc = cand & (w.imag > 1e-9)               # upper half-plane only
    real = cand & (np.abs(w.imag) < 1e-9)

    def pick(mask, score):
        idx = np.where(mask)[0]
        return idx[int(np.argmax(score[idx]))] if idx.size else None

    def info(i):
        return {'rb': Pm[VELRATE, i].sum(), 'struct': 1.0 - Pm[:12, i].sum(),
                'cond': cond[i], 'defective': cond[i] < 1e-3,
                'q': Pm[Q, i], 'p': Pm[P, i], 'w': Pm[WZ, i]}

    modes = {}
    sp = pick(osc, Pm[WZ, :] + Pm[Q, :])       # short-period: heave+pitch-rate
    if sp is not None:
        modes['Short-period'] = (w[sp], info(sp))
    dr = pick(osc, Pm[VY, :] + Pm[R, :])       # Dutch roll: sideslip+yaw-rate
    if dr is not None:
        modes['Dutch roll'] = (w[dr], info(dr))
    roll = pick(cand, Pm[P, :])                # roll: global max roll-rate
    if roll is not None:
        d = info(roll); d['ipr_p'] = inverse_participation_ratio(Pm[P, :])
        modes['Roll subsidence'] = (w[roll], d)
    ph = pick(real, Pm[U, :])                  # phugoid: real, axial-velocity
    if ph is not None:
        modes['Phugoid'] = (w[ph], info(ph))
    spi = pick(real, Pm[PHI, :])               # spiral: real, bank-angle
    if spi is not None:
        modes['Spiral'] = (w[spi], info(spi))
    return modes, w, Pm


def report_modes(name, modes):
    print(f'\n{name}  (participation-factor identified):')
    for nm, (lam, d) in modes.items():
        zeta = -lam.real / abs(lam) if abs(lam) > 0 else float('nan')
        extra = f", IPR_p={d['ipr_p']:.2f}" if 'ipr_p' in d else ''
        extra += f", struct={d['struct']:.2f}" if d['struct'] > 0.1 else ''
        flag = (f"  [ill-conditioned eigvec, cond={d['cond']:.1e}: eigenvalue "
                f"robust, participation caveated]" if d['defective'] else '')
        print(f"  {nm:16s} {lam.real:+8.4f}{lam.imag:+8.4f}j  "
              f"(zeta={zeta:+.2f}, rb={d['rb']:.2f}, q={d['q']:.2f}, "
              f"p={d['p']:.2f}{extra}){flag}")


# Independent verification baseline: the bare 12-state rigid-body system (tail
# aero only, NO wing feedback). Its five modes are textbook-unambiguous; the
# coupled labels are checked against it. Because it omits wing aero, wing-damped
# modes (roll L_p, short-period Z_w) deliberately differ -- that gap measures the
# wing's contribution. See CONTEXT.md "Isolated rigid-body block".
A_iso = np.linalg.inv(Rigid_Dof['M_rr']) @ Rigid_Dof['A_rr']
iso_modes, _, _ = identify_rigid_body_modes(A_iso)
report_modes('Isolated 12-state baseline (tail only)', iso_modes)

rb_modes, _, _ = identify_rigid_body_modes(Rigid_AC_Quasi_Aero['A_Fdyn'])
report_modes('Rigid aircraft (40 states)', rb_modes)

flex_modes, w_flex, Pm_flex = identify_rigid_body_modes(Flexible_AC_Un_Aero['A_Fdyn'])
report_modes('Flexible aircraft (180 states)', flex_modes)

# Where did the rigid pitch/roll content go in the flexible aircraft? List the
# eigenvalues that still carry significant pitch-rate (q) and roll-rate (p).
print('\nFlexible AC: modes retaining rigid q (pitch) > 0.10:')
for i in np.where((Pm_flex[Q, :] > 0.10) & (w_flex.imag >= -1e-9))[0]:
    print(f'   {w_flex[i].real:+8.4f}{w_flex[i].imag:+8.4f}j  '
          f'q={Pm_flex[Q, i]:.2f}  w={Pm_flex[WZ, i]:.2f}  '
          f'struct={1 - Pm_flex[:12, i].sum():.2f}')
print('Flexible AC: roll-rate (p) spread, IPR_p = '
      f'{inverse_participation_ratio(Pm_flex[P, :]):.2f}  '
      f'(rigid AC IPR_p = '
      f'{inverse_participation_ratio(participation_factors(Rigid_AC_Quasi_Aero["A_Fdyn"])[1][P, :]):.2f})')

# Left/right symmetry of the flexible wing modes via FULL nodal-block cosine
# correlation (not a single node's sign): right-wing velocity+displacement block
# is [12:68), left-wing [96:152). corr ~ +1 symmetric, ~ -1 antisymmetric.
# NB: eigenvalues and eigenvectors MUST come from the same decomposition --
# np.linalg.eig and scipy.linalg.eig order their output differently.
wv, Vf = np.linalg.eig(Flexible_AC_Un_Aero['A_Fdyn'])
RW, LW = slice(12, 68), slice(96, 152)
struct_frac = (np.abs(Vf[RW, :]).sum(0) + np.abs(Vf[LW, :]).sum(0)) / \
              (np.abs(Vf).sum(0) + 1e-30)
print('\nFlexible AC: L/R symmetry of near-origin wing modes (full-block corr):')
for i in np.where((struct_frac > 0.3) & (wv.real > -6) &
                  (wv.imag >= -1e-9) & (np.abs(wv.imag) < 6))[0]:
    a, b = Vf[RW, i], Vf[LW, i]
    corr = np.real(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    tag = 'SYM' if corr > 0.3 else ('ANTI' if corr < -0.3 else 'mixed')
    print(f'   {wv[i].real:+8.4f}{wv[i].imag:+8.4f}j  corr={corr:+.3f}  {tag}')


# ---------------------------------------------------------------------------
# Figure: full spectrum (left) + flight-dynamics zoom with mode labels (right)
# ---------------------------------------------------------------------------
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.5))

for ax in (ax_full, ax_zoom):
    ax.plot(np.real(eig_flex), np.imag(eig_flex), 'b*', ms=6,
            label='Flexible AC, unsteady aero (180 states)')
    ax.plot(np.real(eig_rigid), np.imag(eig_rigid), 'ro', mfc='none', ms=7,
            label='Rigid AC, quasi-steady aero (40 states)')
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.axhline(0, color='k', lw=0.8, ls='-', alpha=0.3)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('Real part  [1/s]')

ax_full.set_ylabel('Imaginary part  [rad/s]')
ax_full.set_title('(a) Full eigenvalue spectrum')
ax_full.legend(loc='upper left', fontsize=8)

# Zoom to the near-origin flight-dynamics region. The y-range reaches above the
# rigid short-period (|Im|~2) so the *flexible* short-period (which flexibility +
# unsteady aero pushes up to |Im|~4.7) stays inside the window.
ax_zoom.set_xlim(-6, 1)
ax_zoom.set_ylim(-5.5, 5.5)
ax_zoom.set_title('(b) Flight-dynamics region (rigid-body modes)')

# Annotate the rigid-aircraft rigid-body modes (reference labels). Modes inside
# the zoom window are labelled there with a small point-offset. Modes outside it
# (the fast roll subsidence at Re ~ -49) are labelled on the full-spectrum panel
# with the text parked in empty space, well clear of the near-origin cluster.
offsets = {
    'Short-period': (35, 8),
    'Phugoid':      (8, 16),
    'Dutch roll':   (-16, 10),
    'Spiral':       (12, -18),
}
full_panel_text = {   # data-coordinate text positions on panel (a)
    'Roll subsidence': (-260, 13000),
}
zx0, zx1 = ax_zoom.get_xlim()
zy0, zy1 = ax_zoom.get_ylim()
for name, (lam, _info) in rb_modes.items():
    y = abs(lam.imag)
    in_zoom = (zx0 <= lam.real <= zx1) and (zy0 <= y <= zy1)
    if in_zoom:
        dx, dy = offsets.get(name, (8, 8))
        ax_zoom.annotate(name, xy=(lam.real, y),
                         xytext=(dx, dy), textcoords='offset points',
                         fontsize=8, color='darkred',
                         arrowprops=dict(arrowstyle='->', color='darkred', lw=0.7))
    else:
        tx, ty = full_panel_text.get(name, (lam.real, y))
        ax_full.annotate(name, xy=(lam.real, y), xytext=(tx, ty),
                         textcoords='data', fontsize=8, color='darkred',
                         ha='center',
                         arrowprops=dict(arrowstyle='->', color='darkred', lw=0.7))

# Flexible wing modes inside the zoom window. Identify the structural (bending)
# modes by their participation in the nodal wing DOFs (both wings) and label the
# two that fall in the flight-dynamics region: a real (overdamped) symmetric
# bending root and an oscillatory symmetric bending pair.
def _flex_wing_modes(A, xlim, ylim):
    w, V = np.linalg.eig(A)
    Vn = np.abs(V) / (np.abs(V).sum(axis=0, keepdims=True) + 1e-30)
    # nodal-DOF (velocity+displacement) blocks for right and left wings
    struct = sum(Vn[b:b + 28, :].sum(axis=0) for b in (12, 40, 96, 124))
    inbox = ((w.real >= xlim[0]) & (w.real <= xlim[1]) &
             (np.abs(w.imag) <= ylim[1]) & (struct > 0.5))
    real_b = inbox & (np.abs(w.imag) < 1e-6)
    osc_b = inbox & (w.imag > 1e-6)
    real_mode = w[real_b][np.argmax(struct[real_b])] if real_b.any() else None
    osc_mode = w[osc_b][np.argmax(struct[osc_b])] if osc_b.any() else None
    return real_mode, osc_mode

real_bend, osc_bend = _flex_wing_modes(Flexible_AC_Un_Aero['A_Fdyn'],
                                       ax_zoom.get_xlim(), ax_zoom.get_ylim())

# Arrows: the rigid short-period SPLITS across the two symmetric wing-bending
# modes in the flexible aircraft -- its heave content goes to the oscillatory
# pair (w-participation 0.46 there) and its pitch content predominantly to the
# overdamped real root (q-participation 0.46 there). Both arrows are drawn so the
# figure shows a bifurcation, not a one-to-one merge.
if 'Short-period' in rb_modes:
    sp = rb_modes['Short-period'][0]
    if osc_bend is not None:
        ax_zoom.annotate('', xy=(osc_bend.real, abs(osc_bend.imag)),
                         xytext=(sp.real, abs(sp.imag)),
                         arrowprops=dict(arrowstyle='-|>', color='0.45', lw=1.4,
                                         connectionstyle='arc3,rad=0.25'))
    if real_bend is not None:
        ax_zoom.annotate('', xy=(real_bend.real, 0.0),
                         xytext=(sp.real, abs(sp.imag)),
                         arrowprops=dict(arrowstyle='-|>', color='0.45', lw=1.4,
                                         connectionstyle='arc3,rad=-0.25'))
    ax_zoom.text(-3.8, 3.7, 'rigid short-period splits:\nheave -> osc. sym. bending,\n'
                 'pitch -> overdamped sym.\nbending',
                 fontsize=7, color='0.35', ha='center', va='center', style='italic')

# Blue labels for the flexible symmetric wing-bending modes.
if osc_bend is not None:
    ax_zoom.annotate('1st sym. wing\nbending', xy=(osc_bend.real, abs(osc_bend.imag)),
                     xytext=(10, -2), textcoords='offset points',
                     fontsize=8, color='blue',
                     arrowprops=dict(arrowstyle='->', color='blue', lw=0.7))
if real_bend is not None:
    ax_zoom.annotate('sym. wing bending\n(overdamped)', xy=(real_bend.real, 0.0),
                     xytext=(-30, -34), textcoords='offset points',
                     fontsize=8, color='blue', ha='center',
                     arrowprops=dict(arrowstyle='->', color='blue', lw=0.7))

fig.suptitle('Eigenvalues of the free-flying aircraft: rigid vs. flexible',
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))

# Save the report figure (vector PDF).
fig_path = os.path.join(os.path.dirname(__file__),
                        '..', '..', 'report', 'figures',
                        'partB_aircraft_eigenvalues.pdf')
fig_path = os.path.abspath(fig_path)
fig.savefig(fig_path, bbox_inches='tight')
print(f'Saved figure to {fig_path}')

plt.show()

# Save models
# np.savez('AC_Fdyn_Sys_SS', Rigid_AC_Quasi_Aero=Rigid_AC_Quasi_Aero, Flexible_AC_Un_Aero=Flexible_AC_Un_Aero)
