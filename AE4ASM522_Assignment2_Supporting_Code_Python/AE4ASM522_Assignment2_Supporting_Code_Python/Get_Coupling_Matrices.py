import numpy as np

def Get_Coupling_Matrices(par, flow):
    # Get the number of elements
    Ne = par['N']['Ne']
    
    # Transfer the right wing aerodynamic outputs to flight dynamic body axes
    R_F_rw = np.zeros((3, 3))
    R_F_rw[0, 0] = -2 * par['kD'] * par['Clalpha_W'] * par['alpha_trim']
    R_F_rw[2, 0] = -1
    
    R_M_rw = np.zeros((3, 3))
    R_M_rw[0, 1] = -1
    arm_cg_ec_rw = par['r_cg_rw'][0, 0] - 0.25 * par['chord'][0]
    R_M_rw[1, 0] = arm_cg_ec_rw
    R_M_rw[1, 2] = 1
    
    R_FM_rw = np.vstack((np.zeros((3, 3)), R_F_rw, np.zeros((3, 3)), R_M_rw))
    
    # Transfer the left wing aerodynamic outputs to flight dynamic body axes
    R_F_lw = np.zeros((3, 3))
    R_F_lw[0, 0] = -2 * par['kD'] * par['Clalpha_W'] * par['alpha_trim']
    R_F_lw[2, 0] = -1

    R_M_lw = np.zeros((3, 3))
    R_M_lw[0, 1] = 1  # roll sign flipped vs right wing
    arm_cg_ec_lw = par['r_cg_lw'][0, 0] - 0.25 * par['chord'][0]
    R_M_lw[1, 0] = arm_cg_ec_lw
    R_M_lw[1, 2] = 1

    R_FM_lw = np.vstack((np.zeros((3, 3)), R_F_lw, np.zeros((3, 3)), R_M_lw))
    
    # Transfer the rigid body states to local wing AOA
    R_AOA_rw = np.zeros((Ne, 12))
    R_AOA_rw[:, 5] = np.ones(Ne) / flow['Vel']
    R_AOA_rw[:, 9] = par['r_cg_rw'][1, :] / flow['Vel']
    R_AOA_rw[:, 10] = -par['r_cg_rw'][0, :] / flow['Vel']
    
    R_AOA_lw = np.zeros((Ne, 12))
    R_AOA_lw[:, 5] = np.ones(Ne) / flow['Vel']
    R_AOA_lw[:, 9] = par['r_cg_lw'][1, :] / flow['Vel']
    R_AOA_lw[:, 10] = -par['r_cg_lw'][0, :] / flow['Vel']
    
    # Output
    Coup = {
        'R_FM_rw': R_FM_rw,
        'R_FM_lw': R_FM_lw,
        'R_AOA_rw': R_AOA_rw,
        'R_AOA_lw': R_AOA_lw
    }
    
    return Coup
