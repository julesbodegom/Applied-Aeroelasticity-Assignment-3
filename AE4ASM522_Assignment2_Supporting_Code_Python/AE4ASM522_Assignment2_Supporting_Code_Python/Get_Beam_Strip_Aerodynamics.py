import numpy as np
import scipy
from Get_2D_Aerodynamics import Get_2D_Aerodynamics

def Get_Beam_Strip_Aerodynamics(geo, flow, N):
    # Get_Beam_Strip_Aerodynamics
    # Get the unsteady aerodynamic matrices for a clamped beam using strip theory
    # Each node has four Dofs: heave, bending, torsion, flap (w,phi,theta,beta)

    ## Degrees of freedom
    Nn = N["Nn"]   # number of nodes
    Ne = N["Ne"]   # number of elements
    Nv = N["Nv"]   # number of \dot x_e
    Nd = N["Nd"]   # number of x_e (w,phi,theta,beta)
    Nz = N["Nz"]   # number of lag states (two for Dof, two for gusts)

    ## selection of Dof
    EFT_aero = np.zeros((Nn, 3), dtype=int)

    for i in range(Nn):
        EFT_aero[i,:] = i*4 + np.array([1, 3, 4]) -1

    Ma_non  = np.zeros((Nv, Nv))
    Ca_non  = np.zeros((Nv, Nv))
    Ka_non  = np.zeros((Nd, Nd))
    Ca_cir  = np.zeros((Nv, Nv))
    Ka_cir  = np.zeros((Nd, Nd))
    K_lag   = np.zeros((Nv, Nz))
    A_lag   = np.zeros((Nz, Nz))
    AC_lag  = np.zeros((Nz, Nv))
    AK_lag  = np.zeros((Nz, Nd))
    K_AOA_r = np.zeros((Nv, Nn))  # direct lift (contributed by 0.5 alpha_rigid)

    # correct the lift slope (optional)
    aero_data = scipy.io.loadmat('aero_coeff.mat')
    coe = aero_data['Clalpha_W'] / (2 * np.pi)
    # coe = 1

    for i in range(Nn):
        geo_strip = {
            "b": geo["b"][i],
            "a": geo["a"][i],
            "c": geo["c"][i]
        }
        Aero_strip = Get_2D_Aerodynamics(geo_strip, flow)

        if i == 0 and i == Nn-1:
            lel = geo["L"][i] / 2
        else:
            lel = geo["L"][i]

        Ma_non[np.ix_(EFT_aero[i, :], EFT_aero[i, :])] = coe * lel * Aero_strip["Ma_non"]
        Ca_non[np.ix_(EFT_aero[i, :], EFT_aero[i, :])] = coe * lel * Aero_strip["Ca_non"]
        Ka_non[np.ix_(EFT_aero[i, :], EFT_aero[i, :])] = coe * lel * Aero_strip["Ka_non"]

        Ca_cir[np.ix_(EFT_aero[i, :], EFT_aero[i, :])] = coe * lel * Aero_strip["Ca_cir"]
        Ka_cir[np.ix_(EFT_aero[i, :], EFT_aero[i, :])] = coe * lel * Aero_strip["Ka_cir"]

        K_lag[np.ix_(EFT_aero[i, :], (i)*4 + np.arange(1, 5)-1)]  = coe * lel * Aero_strip["K_lag"]

        K_AOA_r[np.ix_(EFT_aero[i, :]), i] = coe * lel * Aero_strip["K_AOA_r"].T

        A_lag[np.ix_((i)*4 + np.arange(1, 5)-1, (i)*4 + np.arange(1, 5)-1)]  = Aero_strip["A_lag"]
        AC_lag[np.ix_((i)*4 + np.arange(1, 5)-1, EFT_aero[i, :])] = Aero_strip["AC_lag"]
        AK_lag[np.ix_((i)*4 + np.arange(1, 5)-1, EFT_aero[i, :])] = Aero_strip["AK_lag"]

    ## Outputs
    Aero = {
        "Ma_non": Ma_non,
        "Ca_non": Ca_non,
        "Ka_non": Ka_non,
        "Ca_cir": Ca_cir,
        "Ka_cir": Ka_cir,
        "K_lag": K_lag,
        "A_lag": A_lag,
        "AC_lag": AC_lag,
        "AK_lag": AK_lag,
        "K_AOA_r": K_AOA_r
    }

    return Aero

