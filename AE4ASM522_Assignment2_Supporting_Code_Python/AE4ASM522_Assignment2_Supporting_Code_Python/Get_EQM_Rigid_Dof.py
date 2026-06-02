import numpy as np

def Get_EQM_Rigid_Dof(par, flow, Jco_Tail):
    # Get_State_Space model for rigid-body degrees of freedom
    # Assume small perturbations around the trimming condition
    # states are perturbed states around the trimming point
    # states: [Px,Py,Pz|u,v,w|phi,theta,psi|p,q,r|]  12 states
    
    def skew(vector):
        matrix = np.array([[0, -vector[2], vector[1]],
                        [vector[2], 0, -vector[0]],
                        [-vector[1], vector[0], 0]])
        return matrix

    # trimmed condition
    u_trim = np.mean(flow['Vel'] * np.cos(par['alpha_trim']))
    w_trim = np.mean(flow['Vel'] * np.sin(par['alpha_trim']))
    theta_trim = np.mean(par['alpha_trim'])

    # total mass matrix on the left-hand side
    Zero_3 = np.zeros((3, 3))
    I_3 = np.eye(3)

    M_rr = np.block([
        [I_3, Zero_3, Zero_3, Zero_3],
        [Zero_3, par['tot_mass'] * I_3, Zero_3, Zero_3],
        [Zero_3, Zero_3, I_3, Zero_3],
        [Zero_3, Zero_3, Zero_3, par['J_r']]
    ])

    # rigid-body dynamic matrix
    A_dyn = np.zeros((12, 12))
    A_dyn[0:3, 3:6] = np.array([[np.cos(theta_trim), 0, np.sin(theta_trim)],
                                [0, 1, 0],
                                [-np.sin(theta_trim), 0, np.cos(theta_trim)]])
    A_dyn[0:3, 6:9] = np.array([[0, np.cos(theta_trim) * w_trim - np.sin(theta_trim) * u_trim, 0],
                                [-w_trim, 0, np.cos(theta_trim) * u_trim + np.sin(theta_trim) * w_trim],
                                [0, -np.cos(theta_trim) * u_trim - np.sin(theta_trim) * w_trim, 0]])
    skew_matrix = np.mean(par['tot_mass']) * skew([u_trim, 0, w_trim])
    A_dyn[3:6, 9:12] = skew_matrix
    A_dyn[6:9, 9:12] = np.array([[1, 0, np.mean(np.tan(par['alpha_trim']))],
                                 [0, 1, 0],
                                 [0, 0, 1 / np.mean(np.cos(par['alpha_trim']))]])
    A_dyn[3:6, 6:9] = np.mean(par['tot_mass']) * np.mean(par['g']) * np.array([[0, -np.cos(theta_trim), 0],
                                                              [np.cos(theta_trim), 0, 0],
                                                              [0, -np.sin(theta_trim), 0]])
    par_Fx_u = -flow['rho'] * flow['Vel'] * (par['S_fuselage'] + par['S_wing']) * par['C_D0']
    A_dyn[3, 3] = par_Fx_u

    # Aerodynamic influences from the tail
    A_aero_tail = np.zeros((12, 12))
    A_aero_tail[3:6, 3:6] = Jco_Tail[0:3, 0:3]
    A_aero_tail[3:6, 9:12] = Jco_Tail[0:3, 3:6]
    A_aero_tail[9:12, 3:6] = Jco_Tail[3:6, 0:3]
    A_aero_tail[9:12, 9:12] = Jco_Tail[3:6, 3:6]

    # control surface influences [delta_a, delta_e, delta_r]
    B_aero_tail = np.zeros((12, 3))
    B_aero_tail[3:6, :] = Jco_Tail[0:3, 6:9]
    B_aero_tail[9:12, :] = Jco_Tail[3:6, 6:9]

    # assemble
    # M_rr * x_rr_dot = A_rr * x_rr + B_rr * [delta_a, delta_e, delta_r] + [0, Delta F, 0, Delta M]_wing

    A_rr = A_dyn + A_aero_tail
    B_rr = B_aero_tail

    # skew symmetric matrix of a vector
    def skew(vector):
        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])

    # output
    Rigid_Dof = {'M_rr': M_rr, 'A_rr': A_rr, 'B_rr': B_rr}

    return Rigid_Dof
