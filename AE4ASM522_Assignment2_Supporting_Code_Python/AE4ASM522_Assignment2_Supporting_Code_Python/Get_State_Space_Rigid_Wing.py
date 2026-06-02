import numpy as np
from scipy.linalg import inv

def Get_State_Space_Rigid_Wing(Stru, Aero, N, geo, aero_option):
    # Degrees of freedom
    Nn = N['Nn']  # number of nodes
    Ne = N['Ne']  # number of elements
    Nv = N['Nv']  # number of \dot q
    Nd = N['Nd']  # number of q (w,phi,theta,beta)
    Nz = N['Nz']  # number for lag states (two for Dof, two for gusts)

    Ndof  = N['Nv'] + N['Nd'] + N['Nz']

    fxdof = [1, 2, 3, 4, N['Nv'] + 1, N['Nv'] + 2, N['Nv'] + 3, N['Nv'] + 4, N['Nv'] + N['Nd'] + 1, N['Nv'] + N['Nd'] + 2, N['Nv'] + N['Nd'] + 3, N['Nv'] + N['Nd'] + 4]

    ind_beta  = np.arange(4, N['Nd'], 4)-1  # dof for flap

    # Geometric parameters
    lel = geo['Ltot'] / Ne * np.ones(Ne)  # length of element

    # Assemble
    if aero_option == 0:  # unsteady aerodynamics
        # A matrix
        M_ae = Stru['Ms'] - Aero['Ma_non']
        M_ae = M_ae[4:, 4:]  # delete the 0 node Dof
        M_rigid = M_ae[np.ix_(ind_beta, ind_beta)]  # only keep the Dof for flap
        C_ae_un = Stru['Cs'] - Aero['Ca_non'] - Aero['Ca_cir']
        C_ae_un = C_ae_un[4:, 4:]
        C_rigid_un = C_ae_un[np.ix_(ind_beta, ind_beta)]  # only keep the Dof for flap
        K_ae_un = Stru['Ks'] - Aero['Ka_non'] - Aero['Ka_cir']
        K_ae_un = K_ae_un[4:, 4:]
        K_rigid_un = K_ae_un[np.ix_(ind_beta, ind_beta)]  # only keep the Dof for flap

        K_lag = Aero['K_lag'][4:, 4:]
        K_rigid_lag = K_lag[ind_beta, :]  # only keep the Dof for flap

        # AC_lag, AK_lag and A_lag are taken from index 5 onwards.
        AC_lag = Aero["AC_lag"][4:, 4:]
        AC_rigid_lag = AC_lag[:, ind_beta]
        AK_lag = Aero["AK_lag"][4:, 4:]
        AK_rigid_lag = AK_lag[:, ind_beta]
        A_lag = Aero["A_lag"][4:, 4:]
        A_rigid_lag = A_lag[:, :]

        A_rigid_lag_tot = np.hstack([AC_rigid_lag, AK_rigid_lag, A_rigid_lag])

        # The A_rigid matrix is created using np.block() for block matrices
        A_rigid = np.block([
            [-np.linalg.solve(M_rigid, C_rigid_un), -np.linalg.solve(M_rigid, K_rigid_un), np.linalg.solve(M_rigid, K_rigid_lag)],
            [np.eye(N["Ne"], N["Ne"]), np.zeros((N["Ne"], N["Ne"])), np.zeros((N["Ne"], Nz-4))],
            [A_rigid_lag_tot]
        ])

        # Gust AOA input
        B_g = np.zeros((2*Ne+Nz-4, 1))
        ind_z4_rigid = 2*Ne + ((np.arange(1, N["Nn"])-1)*4+4)-1  # index for the fourth lag states
        B_g[ind_z4_rigid, 0] = 1

        # Flap moment input
        B_input = np.eye(Ne, Ne)
        B_f = np.vstack([np.linalg.solve(M_rigid, B_input), np.zeros((Ne, Ne)), np.zeros((Nz-4, Ne))])

        # Rigid angle of attack input matrix
        Brz = np.zeros((Nv-4, Nn-1))
        Brz[1::4, :] = np.eye(Nn-1)  # input as alpha_rigid on lag states z_2 of each node
        K_AOA_r = Aero["K_AOA_r"][4:, 1:]
        F_rigid_AOA_r = K_AOA_r[ind_beta, :]
        B_r = np.vstack([np.linalg.solve(M_rigid, F_rigid_AOA_r), np.zeros((Ne, Ne)), Brz])

        # C matrix
        Cd = np.hstack([np.zeros((Nd, Nv)), np.eye(Nd), np.zeros((Nd, Nz))])  # select q (contains w,phi,theta,beta), delete \dot q and lag states
        Cd = np.delete(Cd, fxdof[0:4], 0)
        Cd = np.delete(Cd, fxdof, 1)

        # External aerodynamic force output
        C_Fa_1 = np.hstack([Aero["Ca_non"] + Aero["Ca_cir"], Aero["Ka_non"] + Aero["Ka_cir"], Aero["K_lag"]])  # direct forces
        C_Fa_1 = np.delete(C_Fa_1, fxdof, 1)
        C_Fa_1 = np.delete(C_Fa_1, np.arange(4), 0)
        index_Cfa1 = np.hstack([ind_beta, ind_beta + 4*Ne, np.linspace(Nd+Nv-7, Nd+Nv+Nv-12, Nv-4)-1])
        index_Cfa1 = index_Cfa1.astype(int)
        C_Fa_1_rigid = C_Fa_1[:, index_Cfa1]

        C_Fa_2_rigid = Aero["Ma_non"][4:, 4:] @ A_rigid[0:4*Ne, :]

        H_root = np.zeros((3, 4*Ne))
        H_root[0, 0::4] = -1  # total lift
        H_root[1, 0::4] = -np.linspace(1, Ne, Ne) * lel  # total root bending moment
        H_root[2, 2::4] = 1  # total root pitching moment

        C_Fa = H_root @ (C_Fa_1_rigid + C_Fa_2_rigid)  # aerodynamic outputs

        # D matrix
        # direct influence from the flap moment control input because of M_a\ddot X_a
        D_f = H_root @ Aero["Ma_non"][4:, 4:] @ B_f[0:4*Ne, :]
        # direct influence of AOA
        D_r = H_root @ Aero["Ma_non"][4:, 4:] @ B_r[0:4*Ne, :] + H_root @ K_AOA_r

        D_g = H_root @ Aero["Ma_non"][4:, 4:] @ B_g[0:4*Ne, :]

    elif aero_option == 1:  # quasi-steady aerodynamics
        # A matrix
        M_ae = Stru["Ms"] - Aero["Ma_non"]
        M_ae = M_ae[4:, 4:]  # delete the 0 node Dof
        M_rigid = M_ae[np.ix_(ind_beta, ind_beta)]  # only keep the Dof for flap
        C_ae_un = Stru["Cs"] - Aero["Ca_non"] - Aero["Ca_cir"] / 0.5
        C_ae_un = C_ae_un[4:, 4:]
        C_rigid_un = C_ae_un[np.ix_(ind_beta, ind_beta)]  # only keep the Dof for flap
        K_ae_un = Stru["Ks"] - Aero["Ka_non"] - Aero["Ka_cir"] / 0.5
        K_ae_un = K_ae_un[4:, 4:]
        K_rigid_un = K_ae_un[np.ix_(ind_beta, ind_beta)]  # only keep the Dof for flap

        A_rigid = np.block([
            [-np.linalg.solve(M_rigid, C_rigid_un), -np.linalg.solve(M_rigid, K_rigid_un)],
            [np.eye(N["Ne"]), np.zeros((N["Ne"], N["Ne"]))]])

        # flap moment input
        B_input = np.eye(Ne, Ne)
        B_f = np.vstack([np.linalg.solve(M_rigid, B_input), np.zeros((Ne, Ne))])

        # rigid angle of attack input matrix
        K_AOA_r = Aero["K_AOA_r"][4:, 1:]
        F_rigid_AOA_r = K_AOA_r[ind_beta, :]
        B_r = np.vstack([np.linalg.solve(M_rigid, F_rigid_AOA_r / 0.5), np.zeros((Ne, Ne))])

        # gust AOA input --> the same influence as rigid AOA in quasi-steady
        B_g = B_r

        # C matrix
        # External aerodynamic force output
        C_Fa_1 = np.hstack([Aero["Ca_non"] + Aero["Ca_cir"] / 0.5, Aero["Ka_non"] + Aero["Ka_cir"] / 0.5])
        C_Fa_1 = np.delete(C_Fa_1, np.s_[fxdof[0:2*4]], axis=1)
        C_Fa_1 = np.delete(C_Fa_1, np.s_[0:4], axis=0)
        C_Fa_1_rigid = C_Fa_1[:, np.hstack([ind_beta, ind_beta + 4 * Ne])]

        M_acc_rigid = Aero["Ma_non"][4:, 4:]
        M_acc_rigid = M_acc_rigid[:, ind_beta]
        C_Fa_2_rigid = M_acc_rigid @ A_rigid[0:Ne, :]  # due to M_a\ddot X_a

        # integrate to the root, force, bending, pitching
        H_root = np.zeros((3, 4 * Ne))
        H_root[0, 0::4] = -1  # total lift
        H_root[1, 0::4] = -np.linspace(1, Ne, Ne) * lel  # total root bending moment
        H_root[2, 2::4] = 1  # total root pitching moment
        
        C_Fa = H_root @ (C_Fa_1_rigid + C_Fa_2_rigid)  # aerodynamic outputs

        # D matrix
        # direct influence from the flap moment control input
        # because of M_a\ddot X_a
        D_f = H_root @ M_acc_rigid @ B_f[0:Ne, :]

        # direct influence from the rigid body AOA input
        D_r = H_root @ M_acc_rigid @ B_r[0:Ne, :] + H_root @ K_AOA_r / 0.5

        # direct influence from the gust input
        D_g = D_r

    # outputs
    Rigid_Sym_SS = {
        "A_rigid": A_rigid,
        "B_g": B_g,
        "B_f": B_f,
        "B_r": B_r,
        "C_Fa": C_Fa,
        "D_f": D_f,
        "D_r": D_r,
        "D_g": D_g
    }
    return Rigid_Sym_SS


    

