import os
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
# Rigid-body (flight-dynamic) mode identification, by eigenvector
# ---------------------------------------------------------------------------
# Body-axis velocity/rate states in the 12-state rigid-body block (0-based):
#   [Px Py Pz | u v w | phi theta psi | p q r]
#   u=3  v=4  w=5  p=9  q=10  r=11   (the dynamic discriminants)
# The classical aircraft modes are labelled from which of these dominate each
# eigenvector. See CONTEXT.md "Rigid-body (flight-dynamic) modes".

def identify_rigid_body_modes(A):
    """Return {mode_name: eigenvalue} for the five classical modes of a
    free-flying aircraft, identified from the dominant body velocity/rate state
    in each eigenvector. The phugoid may be overdamped (two real roots); the
    representative (least-damped) root is returned."""
    w, V = np.linalg.eig(A)
    Vn = np.abs(V) / (np.abs(V).sum(axis=0, keepdims=True) + 1e-30)

    u, v, wv, p, q, r = 3, 4, 5, 9, 10, 11
    phi_a, psi_a = 6, 8                       # bank / heading angles (spiral)
    vel_rate = [u, v, wv, p, q, r]

    # Participation of body velocity/rate states in each eigenvector.
    rb_frac = Vn[vel_rate, :].sum(axis=0)
    integ = np.abs(w) < 1e-6                  # kinematic integrators (~0)

    modes = {}

    def best(mask, key):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        return idx[int(np.argmax([key(i) for i in idx]))]

    osc = (np.abs(w.imag) > 1e-6) & (rb_frac > 0.05)

    # Short-period: oscillatory, w/q dominated.
    sp = best(osc & (Vn[wv, :] + Vn[q, :] > Vn[v, :] + Vn[r, :]),
              key=lambda i: Vn[wv, i] + Vn[q, i])
    if sp is not None:
        modes['Short-period'] = w[sp]

    # Dutch roll: oscillatory, v/r dominated.
    dr = best(osc & (Vn[v, :] + Vn[r, :] >= Vn[wv, :] + Vn[q, :]),
              key=lambda i: Vn[v, i] + Vn[r, i])
    if dr is not None:
        modes['Dutch roll'] = w[dr]

    # Roll subsidence: real, fast (large |Re|), p dominated.
    roll = best((np.abs(w.imag) < 1e-6) & ~integ & (rb_frac > 0.02)
                & (np.argmax(Vn[vel_rate, :], axis=0) == vel_rate.index(p)),
                key=lambda i: -w[i].real)        # most negative real part
    if roll is not None:
        modes['Roll subsidence'] = w[roll]

    # Phugoid: low-frequency longitudinal, u dominated (real or complex).
    # Pick the least-damped (largest real part) u-dominated, slow root that is
    # not the roll mode.
    u_dom = (np.argmax(Vn[vel_rate, :], axis=0) == vel_rate.index(u)) & ~integ
    ph = best(u_dom & (rb_frac > 0.05) & (np.abs(w.real) < 5),
              key=lambda i: w[i].real)
    if ph is not None:
        modes['Phugoid'] = w[ph]

    # Spiral: slow real mode near the origin, lateral (bank/heading) dominated,
    # excluding the pure integrators. Here it is the mildly unstable root.
    lateral_angle = (Vn[phi_a, :] + Vn[psi_a, :] > Vn[vel_rate, :].sum(axis=0))
    sp_cand = (np.abs(w.imag) < 1e-6) & ~integ & lateral_angle & (np.abs(w.real) < 1)
    spi = best(sp_cand, key=lambda i: w[i].real)   # most positive (unstable)
    if spi is not None:
        modes['Spiral'] = w[spi]

    return modes


rb_modes = identify_rigid_body_modes(Rigid_AC_Quasi_Aero['A_Fdyn'])
print('Rigid-aircraft rigid-body modes (eigenvector-identified):')
for name, lam in rb_modes.items():
    zeta = -lam.real / abs(lam) if abs(lam) > 0 else float('nan')
    stab = 'UNSTABLE' if lam.real > 0 else 'stable'
    print(f'  {name:16s} {lam.real:+8.4f} {lam.imag:+8.4f}j  '
          f'(|lambda|={abs(lam):7.4f}, zeta={zeta:+.3f}, {stab})')


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
    'Short-period': (12, 8),
    'Phugoid':      (8, 16),
    'Dutch roll':   (-16, 10),
    'Spiral':       (12, -18),
}
full_panel_text = {   # data-coordinate text positions on panel (a)
    'Roll subsidence': (-260, 13000),
}
zx0, zx1 = ax_zoom.get_xlim()
zy0, zy1 = ax_zoom.get_ylim()
for name, lam in rb_modes.items():
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

# Arrow: the rigid short-period has no rigid-body counterpart in the flexible
# aircraft (max rigid pitch participation ~0.01); its heave/pitch content is
# absorbed into the symmetric wing-bending oscillation.
if 'Short-period' in rb_modes and osc_bend is not None:
    sp = rb_modes['Short-period']
    ax_zoom.annotate('', xy=(osc_bend.real, abs(osc_bend.imag)),
                     xytext=(sp.real, abs(sp.imag)),
                     arrowprops=dict(arrowstyle='-|>', color='0.45', lw=1.4,
                                     connectionstyle='arc3,rad=0.25'))
    ax_zoom.text(-5.0, 3.9, 'rigid short-period\nabsorbed (no rigid\ncounterpart)',
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
