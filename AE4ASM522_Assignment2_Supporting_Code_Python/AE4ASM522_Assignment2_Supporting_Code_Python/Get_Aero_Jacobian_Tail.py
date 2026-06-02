import numpy as np

def Get_Aero_Jacobian_Tail(par, flow):
    # Flow parameters
    rho = flow['rho']
    Vel = flow['Vel']
    
    # Aerodynamic derivatives contributed from the tail
    X_u_T = -rho * Vel * par['S_Htail'] * (par['C_D0'] + par['kD'] * par['Clalpha_HT']**2 * par['ht_alpha_trim']**2) \
        - rho * Vel * par['S_Vtail'] * par['C_D0'] \
        + rho * par['S_Htail'] * par['kD'] * par['Clalpha_HT']**2 * par['ht_alpha_trim']**2 / Vel

    X_w_T = -rho * Vel * par['S_Htail'] * par['kD'] * par['Clalpha_HT']**2 * par['ht_alpha_trim']
    X_v_T = 0
    X_p_T = X_v_T * (-par['r_cg_vt'][2])
    X_q_T = X_w_T * (-par['r_cg_ht'][0])
    X_r_T = X_v_T * par['r_cg_vt'][0]

    X_T = np.array([np.mean(X_u_T), X_v_T, np.mean(X_w_T), X_p_T, np.mean(X_q_T), X_r_T, 0, 0, 0])

    Y_u_T = 0
    Y_v_T = -0.5 * rho * Vel * par['S_Vtail'] * par['Clbeta_VT']
    Y_p_T  = Y_v_T * (-par['r_cg_vt'][2])
    Y_r_T  = Y_v_T * par['r_cg_vt'][0]
    Y_dr_T = -0.5 * rho * Vel**2 * par['S_Vtail'] * par['Cl_delta_r']

    Y_T = np.array([Y_u_T, np.mean(Y_v_T), 0, np.mean(Y_p_T), 0, np.mean(Y_r_T), 0, 0, np.mean(Y_dr_T)])

    Z_u_T = -rho * Vel * par['S_Htail'] * (par['Clalpha_HT'] * par['ht_alpha_trim'] +
                                           par['Cl_delta_e'] * par['delta_e_trim'])
    Z_w_T = -0.5 * rho * Vel * par['S_Htail'] * par['Clalpha_HT']
    Z_q_T  = Z_w_T * (-par['r_cg_ht'][0])
    Z_de_T = -0.5 * rho * Vel**2 * par['S_Htail'] * par['Cl_delta_e']

    Z_T = np.array([np.mean(Z_u_T), 0, np.mean(Z_w_T), 0, np.mean(Z_q_T), 0, 0, np.mean(Z_de_T), 0])

    X_T_ht = [np.mean(-rho * Vel * par['S_Htail'] * (par['C_D0'] + par['kD'] * par['Clalpha_HT']**2 * par['ht_alpha_trim']**2)),
              0, np.mean(X_w_T), 0, np.mean(X_q_T), 0, 0, 0, 0]
    X_T_vt = [-rho * Vel * par['S_Vtail'] * par['C_D0'], X_v_T, 0, X_p_T, 0, X_r_T, 0, 0, 0]

    L_T = -par['r_cg_vt'][2] * Y_T
    M_T = par['r_cg_ht'][2] * np.array(X_T_ht) + par['r_cg_vt'][2] * np.array(X_T_vt) - par['r_cg_ht'][0] * Z_T
    N_T = par['r_cg_vt'][0] * Y_T

    Jco_Tail = np.vstack((X_T, Y_T, Z_T, L_T, M_T, N_T))

    return Jco_Tail
