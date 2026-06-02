import numpy as np

def Get_State_Space_Elastic_Wing(Stru, Aero, N, geo, aero_option):
    # Get_State_Space
    # Assemble the aerodynamic and structural matrices to state-space
    # each node has four Dof: heave, bending, torsion, flap (w,phi,theta,beta)
    # heave --> downwards positive 
    # torsion --> nose up positive 
    # bending --> bend up positive 
    # flap --> flap down positive 

    ## Degrees of freedom
    Nn = N['Nn']   # number of nodes
    Ne = N['Ne']   # number of elements
    Nv = N['Nv']   # number of \dot q
    Nd = N['Nd']   # number of q (w,phi,theta,beta)
    Nz = N['Nz']   # number for lag states (two for Dof, two for gusts)

    Ndof  = N['Nv']+N['Nd']+N['Nz']

    fxdof = [0, 1, 2, 3] + list(range(N['Nv'], N['Nv'] + 4)) + list(range(N['Nv'] + N['Nd'], N['Nv'] + N['Nd'] + 4))
    frdof = np.setdiff1d(np.arange(1, Ndof), fxdof)  # the remaining states expect for the clamped states

    ## Geometric parameters
    lel = geo['Ltot']/Ne*np.ones(Ne) # length of element

    ## Assemble
    if aero_option == 0: # unsteady aerodynamics
        ## A matrix
        M_ae = Stru['Ms'] - Aero['Ma_non']
        M_ae = M_ae[4:,4:] # delete the 0 node Dof
        C_ae_un = Stru['Cs']-Aero['Ca_non']-Aero['Ca_cir']
        C_ae_un = C_ae_un[4:,4:]
        K_ae_un = Stru['Ks']-Aero['Ka_non']-Aero['Ka_cir']
        K_ae_un = K_ae_un[4:,4:]
        K_lag = Aero['K_lag'][4:,4:]
        
        A_lag_tot = np.concatenate([Aero['AC_lag'], Aero['AK_lag'], Aero['A_lag']], axis=1)
        
        A_ae = np.block([
            [-np.linalg.solve(M_ae, C_ae_un), -np.linalg.solve(M_ae, K_ae_un), np.linalg.solve(M_ae, K_lag)],
            [np.eye(Nv-4, Nv-4), np.zeros((Nv-4, Nd-4)), np.zeros((Nv-4, Nz-4))],
            [A_lag_tot[4:,frdof]]])
        
        ## gust AOA input  --> influences the fourth lag state  
        B_z4  = np.zeros((Nv-4,Nn-1))
        B_z4[3::4,:] = np.eye(Nn-1)   #input as alpha_rigid on lag states z_2 of each node
        B_g = np.block([[np.zeros((Nv-4, Nn-1))], [np.zeros((Nv-4, Nn-1))], [B_z4]]) 
        
        ## flap moment input
        B_input  = np.zeros((Nv-4,Nn-1))
        B_input[3::4,:] = np.eye(Nn-1)  # select the elements relevant to beta
        B_f = np.block([[np.linalg.solve(M_ae, B_input)], [np.zeros((Nv-4, Nn-1))], [np.zeros((Nv-4, Nn-1))]])
        
        ## rigid angle of attack input matrix
        Brz  = np.zeros((Nv-4,Nn-1))
        Brz[1::4,:] = np.eye(Nn-1)   #input as alpha_rigid on lag states z_2 of each node
        K_AOA_r = Aero['K_AOA_r'][4:,1:]
        B_r = np.block([[np.linalg.solve(M_ae, K_AOA_r)], [np.zeros((Nv-4, Nn-1))], [Brz]])
        
        ## C matrix
        # displacements
        Cd = np.block([[np.zeros((Nd, Nv))], [np.eye(Nd)], [np.zeros((Nd, Nz))]]).T  #select q (contains w,phi,theta,beta)  delete \dot q and lag states
        Cd = np.delete(Cd, fxdof[0:4], axis=0)
        Cd = np.delete(Cd, fxdof, axis=1)

        # External aerodynamic force output, different from the root reaction force!
        C_Fa_1 = np.hstack([Aero['Ca_non'] + Aero['Ca_cir'], Aero['Ka_non'] + Aero['Ka_cir'], Aero['K_lag']])  # direct forces
        C_Fa_1 = np.delete(C_Fa_1, fxdof, axis=1)
        C_Fa_1 = np.delete(C_Fa_1, np.s_[0:4], axis=0)
        C_Fa_2 = Aero['Ma_non'][4:, 4:] @ A_ae[0:4*Ne, :]  # due to M_a\ddot X_a
    
        # integrate to the root, force, bending, pitching
        H_root = np.zeros((3, 4*Ne))
        H_root[0, 0:4*Ne:4] = -1  # total lift
        H_root[1, 0:4*Ne:4] = -np.linspace(1, Ne, Ne) * lel  # total root bending moment
        H_root[2, 2:4*Ne:4] = 1  # total root pitching moment
    
        C_Fa = H_root @ (C_Fa_1 + C_Fa_2)  # aerodynamic outputs

        ## D matrix
        # direct influence from the flap moment control input
        D_f = H_root@Aero['Ma_non'][4:,4:]@B_f[0:4*Ne,:]

        # direct influence from the rigid body AOA input
        D_r = H_root@Aero['Ma_non'][4:,4:]@B_r[0:4*Ne,:] + H_root@K_AOA_r

        # direct influence from the gust input = 0
        D_g = H_root@Aero['Ma_non'][4:,4:]@B_g[0:4*Ne,:]

        #output of the root reaction forces (if needed)
        C_root_1 = np.hstack([Stru['Cs'] - Aero['Ca_non'] - Aero['Ca_cir'], Stru['Ks'] - Aero['Ka_non'] - Aero['Ka_cir'], -Aero['K_lag']])
        C_root_1 = C_root_1[0:4, :]
        C_root_1 = np.delete(C_root_1, fxdof, axis=1)
        Macc = Stru['Ms'] - Aero['Ma_non']
        Macc = Macc[0:4, 4:8]
        C_root_2 = Macc @ A_ae[0:4, :]
        C_root = C_root_1 + C_root_2
    
        D_root_f = Macc @ B_f[0:4, :]
        D_root_r = Macc @ B_r[0:4, :]
    elif aero_option == 1: # quasi-steady aerodynamics
        # A matrix
        M_ae = Stru['Ms'] - Aero['Ma_non']
        M_ae = M_ae[4:, 4:] # delete the 0 node Dof
        C_ae_un = Stru['Cs'] - Aero['Ca_non'] - Aero['Ca_cir'] / 0.5
        C_ae_un = C_ae_un[4:, 4:]
        K_ae_un = Stru['Ks'] - Aero['Ka_non'] - Aero['Ka_cir'] / 0.5
        K_ae_un = K_ae_un[4:, 4:]
        
        A_ae = np.vstack([
            np.hstack([-np.linalg.solve(M_ae, C_ae_un), -np.linalg.solve(M_ae, K_ae_un)]),
            np.hstack([np.eye(Nv - 4, Nv - 4), np.zeros((Nv - 4, Nd - 4))])
        ])
        
        # flap moment input
        B_input  = np.zeros((Nv - 4, Nn - 1))
        B_input[3::4, :] = np.eye(Nn - 1)  # select the elements relavent to beta
        B_f = np.vstack([np.linalg.solve(M_ae, B_input), np.zeros((Nv - 4, Nn - 1))])
        
        # rigid angle of attack input matrix
        K_AOA_r = Aero['K_AOA_r'][4:, 1:]
        B_r = np.vstack([np.linalg.solve(M_ae, K_AOA_r / 0.5), np.zeros((Nv - 4, Nn - 1))])  # all the lift build up instantly
        
        # gust AOA input --> the same influence as rigid AOA in quasi-steady
        B_g = B_r
        
        # C matrix
        # External aerodynamic force output
        C_Fa_1 = np.hstack([Aero['Ca_non'] + Aero['Ca_cir'] / 0.5, Aero['Ka_non'] + Aero['Ka_cir'] / 0.5])
        C_Fa_1 = np.delete(C_Fa_1, np.array(fxdof[:2*4]), axis=1)
        C_Fa_1 = np.delete(C_Fa_1, np.arange(0, 4), axis=0)
        C_Fa_2 = Aero['Ma_non'][4:, 4:] @ A_ae[0:4*Ne, :]  # due to M_a\ddot X_a

        # integrate to the root, force, bending, pitching
        H_root  = np.zeros((3, 4*Ne))
        H_root[0, 0:4*Ne:4] = -1  # total lift
        H_root[1, 0:4*Ne:4] = -np.linspace(1, Ne, Ne) * lel  # total root bending moment
        H_root[2, 2:4*Ne:4] = 1  # total root pitching moment
        
        C_Fa = H_root @ (C_Fa_1 + C_Fa_2)  # aerodynamic outputs
        
        # D matrix
        # direct influence from the flap moment control input
        # because of M_a\ddot X_a
        D_f = H_root @ Aero['Ma_non'][4:, 4:] @ B_f[0:4*Ne, :]
      
        # direct influence from the rigid body AOA input
        D_r = H_root @ (Aero['Ma_non'][4:, 4:] @ B_r[0:4*Ne, :] + K_AOA_r / 0.5)
        
        # direct influence from the gust input
        D_g = D_r

    Ae_Sym_SS = {
        'A_ae': A_ae,
        'B_g': B_g,
        'B_f': B_f,
        'B_r': B_r,
        'C_Fa': C_Fa,
        'D_f': D_f,
        'D_r': D_r,
        'D_g': D_g
    }

    return Ae_Sym_SS
        
