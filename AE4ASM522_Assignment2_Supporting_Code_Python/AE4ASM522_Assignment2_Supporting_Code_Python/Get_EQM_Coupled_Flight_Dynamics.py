import numpy as np
from Get_FEM_Beam_Structure import Get_FEM_Beam_Structure
from Get_Beam_Strip_Aerodynamics import Get_Beam_Strip_Aerodynamics
from Get_State_Space_Elastic_Wing import Get_State_Space_Elastic_Wing
from Get_State_Space_Rigid_Wing import Get_State_Space_Rigid_Wing

def Get_EQM_Coupled_Flight_Dynamics(Rigid_Dof, Coup, flow, par, option):
    # Parameters
    geo = par['geo']
    stiff = par['stiff']
    mass = par['mass']
    N = par['N']

    # Rigid body EQM
    # states: [Px,Py,Pz|u,v,w|phi,theta,psi|p,q,r|]  12 Dof
    # M_rr \dot x_rr = A_rr x_rr + B_rr*[delta_a,delta_e,delta_r] + ...
    # [0,\Delta F,0,\Delta M]_wing

    M_rr = Rigid_Dof['M_rr']
    A_rr = Rigid_Dof['A_rr']
    B_rr = Rigid_Dof['B_rr']

    # Clamped wing EQM
    # states x_w: depends on the selections, can include \dot x_e, x_e, z (lag states)
    # note the EQM for the left and right wings are the same, only the later
    # used transfer matrices are different

    # inputs:
    # alpha_r (rigid body AOA input)
    # alpha_g (gust AOA input)
    # M_f (flap moment input)

    # outputs: [lift, bending moment, torsion moment]  w.r.t to the wing root

    # EQM
    # \dot x_w = A_ww x_w + B_w_r * alpha_r + B_w_g * alpha_g + B_w_f * M_f
    #        y = C_ww x_w + D_w_r * alpha_r + D_w_g * alpha_g + D_w_f * M_f
    
    Stru = Get_FEM_Beam_Structure(geo, stiff, mass, N)
    Aero = Get_Beam_Strip_Aerodynamics(geo, flow, N)

    if option['stru'] == 0:  # Elastic wing
        Ae_Sym_SS = Get_State_Space_Elastic_Wing(Stru, Aero, N, geo, option['aero'])
        A_ww = Ae_Sym_SS['A_ae']
        B_w_r = Ae_Sym_SS['B_r']
        B_w_g = Ae_Sym_SS['B_g']
        B_w_f = Ae_Sym_SS['B_f']
        C_ww = Ae_Sym_SS['C_Fa']
        D_w_r = Ae_Sym_SS['D_r']
        D_w_g = Ae_Sym_SS['D_g']
        D_w_f = Ae_Sym_SS['D_f']
    elif option['stru'] == 1:  # Rigid wing
        Rigid_Sym_SS = Get_State_Space_Rigid_Wing(Stru, Aero, N, geo, option['aero'])
        A_ww = Rigid_Sym_SS['A_rigid']
        B_w_r = Rigid_Sym_SS['B_r']
        B_w_g = Rigid_Sym_SS['B_g']
        B_w_f = Rigid_Sym_SS['B_f']
        C_ww = Rigid_Sym_SS['C_Fa']
        D_w_r = Rigid_Sym_SS['D_r']
        D_w_g = Rigid_Sym_SS['D_g']
        D_w_f = Rigid_Sym_SS['D_f']

    # Couple the rigid and wing aeroelastic Dofs
    A11 = np.linalg.inv(M_rr) @ (A_rr + Coup['R_FM_rw'] @ D_w_r @ Coup['R_AOA_rw'] +
                                 Coup['R_FM_lw'] @ D_w_r @ Coup['R_AOA_lw'])
    A12 = np.linalg.inv(M_rr) @ Coup['R_FM_rw'] @ C_ww
    A13 = np.linalg.inv(M_rr) @ Coup['R_FM_lw'] @ C_ww

    A21 = B_w_r @ Coup['R_AOA_rw']
    A31 = B_w_r @ Coup['R_AOA_lw']

    A_Fdyn = np.block([[A11, A12, A13],
                       [A21, A_ww, np.zeros_like(A_ww)],
                       [A31, np.zeros_like(A_ww), A_ww]])

    B_rr_g = np.concatenate([np.linalg.inv(M_rr) @ Coup['R_FM_rw'] @ D_w_g,
                             np.linalg.inv(M_rr) @ Coup['R_FM_lw'] @ D_w_g], axis=1)
    B_rw_g = np.concatenate([B_w_g, np.zeros_like(B_w_g)], axis=1)
    B_lw_g = np.concatenate([np.zeros_like(B_w_g), B_w_g], axis=1)
    B_Fdyn_gust = np.block([[B_rr_g],
                            [B_rw_g],
                            [B_lw_g]])

    B_rr_f = np.concatenate([np.linalg.inv(M_rr) @ Coup['R_FM_rw'] @ D_w_f,
                             np.linalg.inv(M_rr) @ Coup['R_FM_lw'] @ D_w_f], axis=1)
    B_rw_f = np.concatenate([B_w_f, np.zeros_like(B_w_f)], axis=1)
    B_lw_f = np.concatenate([np.zeros_like(B_w_f), B_w_f], axis=1)
    B_Fdyn_flap = np.block([[B_rr_f],
                            [B_rw_f],
                            [B_lw_f]])

    B_rr_delta = np.linalg.inv(M_rr) @ B_rr
    B_rw_delta = np.zeros((A_ww.shape[0], 3))
    B_lw_delta = np.zeros((A_ww.shape[0], 3))
    B_Fdyn_delta = np.block([[B_rr_delta],
                             [B_rw_delta],
                             [B_lw_delta]])

    # Outputs
    Fdyn_Sys_SS = {
        'A_Fdyn': A_Fdyn,
        'B_Fdyn_gust': B_Fdyn_gust,
        'B_Fdyn_flap': B_Fdyn_flap,
        'B_Fdyn_delta': B_Fdyn_delta
    }

    return Fdyn_Sys_SS

