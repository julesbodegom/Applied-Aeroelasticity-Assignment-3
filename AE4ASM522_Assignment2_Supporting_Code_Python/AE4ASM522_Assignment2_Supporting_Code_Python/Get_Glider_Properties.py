import math
import numpy as np
import scipy.io

def Get_Glider_Properties():
    par = {}
        
    r2d = 180 / math.pi
    d2r = math.pi / 180
    
    # Mass properties
    par['tot_mass'] = 227
    
    Ixx = 493.819
    Iyy = 726.733
    Izz = 1170.472
    Ixy = 0
    Ixz = -34.039
    Iyz = 0
    par['J_r'] = np.array([[Ixx, Ixy, Ixz],
                           [Ixy, Iyy, Iyz],
                           [Ixz, Iyz, Izz]])
    
    par['g'] = 9.81  # gravitational acceleration
    
    # Tail geometry properties
    par['S_wing'] = 9.60
    par['S_Htail'] = 0.84
    par['S_Vtail'] = 1.08
    par['S_half_wing'] = par['S_wing'] / 2
    par['S_fuselage'] = 1.30  # area of fuselage that faces the wind (drag)
    
    # Distances from c.g. to the a.c. of each lifting surface
    cg_shift = 0.424
    
    par['r_cg_ht'] = np.array([-3.7940, 0, -1.2000]) + [cg_shift, 0, 0]
    par['r_cg_vt'] = np.array([-3.9440, 0, -0.5570]) + [cg_shift, 0, 0]
    
    # Wing geometry properties
    par['Ltot'] = 8
    par['Nn'] = 8  # number of nodes
    par['chord'] = 0.6 * np.ones(par['Nn'])
    
    dihedral = 3 * d2r
    r_cg_rw = np.zeros((7, 3))
    r_cg_lw = np.zeros((7, 3))

    r_wing = np.linspace(par['Ltot'] / ((par['Nn'] - 1) * 2), par['Ltot']-(par['Ltot'] / ((par['Nn'] - 1) * 2)), par['Nn'] - 1)  # middle of every strip
    r_cg_rw[:, 0] = np.ones(7) * (-0.3240 + cg_shift)
    r_cg_rw[:, 1] = np.cos(dihedral) * r_wing
    r_cg_rw[:, 2] = -np.sin(dihedral) * r_wing
    par['r_cg_rw'] = r_cg_rw.T
    
    r_cg_lw[:, 0] = np.ones(7) * (-0.3240 + cg_shift)
    r_cg_lw[:, 1] = -np.cos(dihedral) * r_wing
    r_cg_lw[:, 2] = -np.sin(dihedral) * r_wing
    par['r_cg_lw'] = r_cg_lw.T
    
    # Degrees of freedom
    N = {}
    N['Nn'] = par['Nn']  # number of nodes
    N['Ne'] = N['Nn'] - 1  # number of elements
    N['Nv'] = 4 * N['Nn']  # number of \dot q
    N['Nd'] = 4 * N['Nn']  # number of q (w,phi,theta,beta)
    N['Nz'] = 4 * N['Nn']  # number of lag states (two for Dof, two for gusts)
    N['Ndof'] = N['Nv'] + N['Nd'] + N['Nz']
    N['fxdof'] = [1, 2, 3, 4, N['Nv']+1, N['Nv']+2, N['Nv']+3, N['Nv']+4]
    N['frdof'] = list(set(range(1, N['Ndof']+1)) - set(N['fxdof']))
    
    par['N'] = N
    
    # Geometric parameters
    geo = {}
    geo['b'] = par['chord'] / 2  # half chord
    geo['Ltot'] = par['Ltot']  # half span
    
    geo['a'] = np.zeros(N['Nn'])  # Location of the shear centre wrt midchord
    geo['c'] = 0.5 * np.ones(N['Nn'])  # Location of the flap hinge wrt midchord
    geo['xcg'] = 0.1 * geo['b']  # Location of the cg wrt shear centre
    geo['L'] = geo['Ltot'] / N['Ne'] * np.ones(N['Nn'])  # length per element
    geo['Lf'] = (1 - geo['c']) * geo['b']
    
    par['geo'] = geo
    
    # Mass parameters
    mass = {}
    mass['m'] = 0.75 * np.ones(N['Nn'])  # Wing mass per unit length
    mass['It'] = 0.1 * np.ones(N['Nn'])
    mass['mf'] = 0.25 * np.ones(N['Nn'])  # Flap mass per flap
    mass['If_cgf'] = 1e-3 * np.ones(N['Nn'])  # Inertia of the flap around its own cg, per flap
    mass['xf'] = -0.1 * np.ones(N['Nn'])  # Location of the flap cg with respect to the hinge line
    mass['Sf'] = mass['mf'] * mass['xf'] * geo['Lf']
    mass['If'] = mass['If_cgf'] + mass['Sf'] * mass['xf'] * geo['Lf']
    
    par['mass'] = mass
    
    # Stiffness parameters
    stiff = {}
    stiff['E'] = 70e9
    stiff['G'] = stiff['E'] / 2 / (1 + 0.3)
    stiff['Ixx'] = 1.2e5 / stiff['E'] * np.ones(N['Nn'])
    stiff['J'] = 1e7 / stiff['G'] * np.ones(N['Nn'])
    stiff['Kf'] = 1e2 * np.ones(N['Nn'])
    stiff['psic'] = np.zeros(N['Nn'])  # Bending torsion coupling term
    stiff['Kc'] = np.sign(stiff['psic']) * (stiff['psic']**2 * 16 * stiff['E'] * stiff['Ixx'] *
                                            stiff['G'] * stiff['J'] / geo['L']**2)**0.5
    
    par['stiff'] = stiff
    
    # Aerodynamic coefficients
    # .mat scalars load as (1,1) arrays; squeeze to floats so they can be
    # assigned into scalar matrix slots (NumPy >=2.0 is strict about this).
    aero_data = scipy.io.loadmat('aero_coeff.mat')
    par['Clalpha_W'] = float(np.squeeze(aero_data['Clalpha_W']))
    par['Clalpha_HT'] = float(np.squeeze(aero_data['Clalpha_HT']))
    par['Clbeta_VT'] = float(np.squeeze(aero_data['Clbeta_VT']))
    par['Cl_delta_e'] = float(np.squeeze(aero_data['Cl_delta_e']))
    par['Cl_delta_r'] = float(np.squeeze(aero_data['Cl_delta_r']))
    
    # Drag coefficients
    par['kD'] = 0.04  # induced drag coefficient
    par['C_D0'] = 0.5  # frictional drag coefficient
    
    # Trim values
    trim_data = scipy.io.loadmat('trim_V35.mat')
    par['alpha_trim'] = float(np.squeeze(trim_data['alpha_trim']))  # trim angle of the wing
    par['delta_e_trim'] = float(np.squeeze(trim_data['delta_e_trim']))
    par['T_trim'] =  float(np.squeeze(trim_data['T_trim']))
    AOA_install = 0  # install angle for the horizontal tail
    par['ht_alpha_trim'] = par['alpha_trim'] + AOA_install  # trim angle of the wing
    
    return par
