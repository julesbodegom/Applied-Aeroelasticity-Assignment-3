import numpy as np
import matplotlib.pyplot as plt

from Get_2D_Structure import Get_2D_Structure
from Get_2D_Aerodynamics import Get_2D_Aerodynamics


d2r = np.pi / 180
r2d = 180 / np.pi

# Geometric parameters
geo = {'b': 0.25, 'a': -0.2, 'c': 0.5}

# Flow conditions
flow = {'rho': 1.225, 'Vel': 10}

# Get structural mass, damping, and stiffness
Stru = Get_2D_Structure(geo)

# Get aerodynamic matrices
Aero = Get_2D_Aerodynamics(geo, flow)

# Assemble to state-space with unsteady aerodynamics
M_ae = Stru['Ms'] - Aero['Ma_non']
C_ae_un = Stru['Cs'] - Aero['Ca_non'] - Aero['Ca_cir']
K_ae_un = Stru['Ks'] - Aero['Ka_non'] - Aero['Ka_cir']

A_ae_unsteady = np.block([
    [np.linalg.solve(-M_ae, C_ae_un), np.linalg.solve(-M_ae, K_ae_un), np.linalg.solve(M_ae, Aero['K_lag'])],
    [np.eye(3), np.zeros((3,3)), np.zeros((3,4))],
    [Aero['AC_lag'], Aero['AK_lag'], Aero['A_lag']]
])

# Rigid body AOA input
B_r_unsteady = np.zeros((A_ae_unsteady.shape[0], 1))
B_r_unsteady[:3] = np.linalg.solve(M_ae, Aero['K_AOA_r'])
B_r_unsteady[-3] = 1

# Gust AOA input
B_g_unsteady = np.zeros((A_ae_unsteady.shape[0], 1))
B_g_unsteady[-1] = 1

# Flap hinge moment input
B_b_unsteady = np.zeros((A_ae_unsteady.shape[0], 1))
B_b_unsteady[:3] = np.linalg.solve(M_ae, np.array([[0, 0, 1]]).T)

# Assemble to state-space with quasi-steady aerodynamics
C_ae_quasi = Stru['Cs'] - Aero['Ca_non'] - Aero['Ca_cir'] / 0.5
K_ae_quasi = Stru['Ks'] - Aero['Ka_non'] - Aero['Ka_cir'] / 0.5

A_ae_quasi = np.block([
    [np.linalg.solve(-M_ae, C_ae_quasi), np.linalg.solve(-M_ae, K_ae_quasi)],
    [np.eye(3), np.zeros((3, 3))]
])

# Rigid body AOA input
B_r_quasi = np.zeros((A_ae_quasi.shape[0], 1))
B_r_quasi[:3] = np.linalg.solve(M_ae, Aero['K_AOA_r'] / 0.5)

# Gust AOA input
B_g_quasi = np.zeros((A_ae_quasi.shape[0], 1))
B_g_quasi[:3] = np.linalg.solve(M_ae, Aero['K_AOA_r'] / 0.5)

# Flap hinge moment input
B_b_quasi = np.zeros((A_ae_quasi.shape[0], 1))
B_b_quasi[:3] = np.linalg.solve(M_ae, np.array([[0, 0, 1]]).T)

# Trim analysis 
AOA = 5 * d2r
x_trim_un = -np.linalg.solve(A_ae_unsteady, B_r_unsteady * AOA)
x_trim_quasi = -np.linalg.solve(A_ae_quasi, B_r_quasi * AOA)

# Model analysis
poles_A_ae_unsteady = np.linalg.eigvals(A_ae_unsteady)
poles_A_ae_quasi = np.linalg.eigvals(A_ae_quasi)

plt.figure()
plt.plot(np.real(poles_A_ae_unsteady), np.imag(poles_A_ae_unsteady), 'b*')
plt.grid(True)
plt.plot(np.real(poles_A_ae_quasi), np.imag(poles_A_ae_quasi), 'ro')
plt.legend(['unsteady aerodynamics', 'quasi-steady aerodynamics'])
plt.title('Eigenvalues of a 2D typical wing section')
plt.show()
