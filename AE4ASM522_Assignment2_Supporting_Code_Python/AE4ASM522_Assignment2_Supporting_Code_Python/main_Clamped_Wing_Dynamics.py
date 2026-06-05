import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from Get_Beam_Strip_Aerodynamics import Get_Beam_Strip_Aerodynamics
from Get_State_Space_Elastic_Wing import Get_State_Space_Elastic_Wing
from Get_FEM_Beam_Structure import Get_FEM_Beam_Structure
from Get_State_Space_Rigid_Wing import Get_State_Space_Rigid_Wing


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
r2d = 180 / np.pi
d2r = np.pi / 180


# -------------------------------------------------------------------------
# Define N
# -------------------------------------------------------------------------
N = {}
N["Nn"] = 8                     # number of nodes
N["Ne"] = N["Nn"] - 1           # number of elements
N["Nv"] = 4 * N["Nn"]           # number of q_dot
N["Nd"] = 4 * N["Nn"]           # number of q = (w, phi, theta, beta)
N["Nz"] = 4 * N["Nn"]           # number of lag states
N["Ndof"] = N["Nv"] + N["Nd"] + N["Nz"]

N["fxdof"] = (
    [1, 2, 3, 4]
    + list(np.arange(N["Nv"] + 1, N["Nv"] + 5))
    + list(np.arange(N["Nv"] + N["Nd"] + 1, N["Nv"] + N["Nd"] + 5))
)

N["frdof"] = list(set(np.arange(1, N["Ndof"] + 1)) - set(N["fxdof"]))


# -------------------------------------------------------------------------
# Geometry
# -------------------------------------------------------------------------
geo = {}
geo["b"] = 0.6 * np.ones(N["Nn"]) / 2      # half chord
geo["a"] = np.zeros(N["Nn"])               # shear centre wrt midchord
geo["c"] = 0.5 * np.ones(N["Nn"])          # flap hinge wrt midchord
geo["xcg"] = 0.1 * geo["b"]                # cg wrt shear centre
geo["Ltot"] = 8                            # half span
geo["L"] = geo["Ltot"] / N["Ne"] * np.ones(N["Nn"])
geo["Lf"] = (1 - geo["c"]) * geo["b"]


# -------------------------------------------------------------------------
# Mass properties
# -------------------------------------------------------------------------
mass = {}
mass["m"] = 0.75 * np.ones(N["Nn"])
mass["It"] = 0.1 * np.ones(N["Nn"])
mass["mf"] = 0.25 * np.ones(N["Nn"])
mass["If_cgf"] = 1e-3 * np.ones(N["Nn"])
mass["xf"] = -0.1 * np.ones(N["Nn"])
mass["Sf"] = mass["mf"] * mass["xf"] * geo["Lf"]
mass["If"] = mass["If_cgf"] + mass["Sf"] * mass["xf"] * geo["Lf"]


# -------------------------------------------------------------------------
# Stiffness properties
# -------------------------------------------------------------------------
stiff = {}
stiff["E"] = 70e9
stiff["G"] = stiff["E"] / 2 / (1 + 0.3)
stiff["Ixx"] = 1.2e5 / stiff["E"] * np.ones(N["Nn"])
stiff["J"] = 1e7 / stiff["G"] * np.ones(N["Nn"])
stiff["Kf"] = 1e2 * np.ones(N["Nn"])
stiff["psic"] = np.zeros(N["Nn"])
stiff["Kc"] = np.sign(stiff["psic"]) * (
    stiff["psic"]**2
    * 16.0
    * stiff["E"]
    * stiff["Ixx"]
    * stiff["G"]
    * stiff["J"]
    / geo["L"]**2
) ** 0.5


# -------------------------------------------------------------------------
# Flow parameters
# -------------------------------------------------------------------------
flow = {}
flow["rho"] = 1.225
flow["Vel"] = 35


# -------------------------------------------------------------------------
# Get structural and aerodynamic matrices
# -------------------------------------------------------------------------
Stru = Get_FEM_Beam_Structure(geo, stiff, mass, N)
Aero = Get_Beam_Strip_Aerodynamics(geo, flow, N)


# =========================================================================
# Flexible clamped wing beam eigenvalue analysis
# =========================================================================
plt.figure(1)

# Unsteady aerodynamics
aero_option = 0
Ae_Sym_SS = Get_State_Space_Elastic_Wing(Stru, Aero, N, geo, aero_option)
eig_flex_us = np.linalg.eigvals(Ae_Sym_SS["A_ae"])
plt.plot(eig_flex_us.real, eig_flex_us.imag, "b*", label="unsteady aerodynamics")

# Quasi-steady aerodynamics
aero_option = 1
Ae_Sym_SS = Get_State_Space_Elastic_Wing(Stru, Aero, N, geo, aero_option)
eig_flex_qs = np.linalg.eigvals(Ae_Sym_SS["A_ae"])
plt.plot(eig_flex_qs.real, eig_flex_qs.imag, "ro", label="quasi-steady aerodynamics")

plt.axvline(x=0, color="k", linestyle="--", linewidth=0.8)
plt.grid(True)
plt.legend()
plt.xlabel("Real part [1/s]")
plt.ylabel("Imaginary part [rad/s]")
plt.title("Eigenvalues of a flexible clamped wing beam")


# =========================================================================
# Rigid clamped wing beam eigenvalue analysis
# =========================================================================
plt.figure(2)

# Unsteady aerodynamics
aero_option = 0
Rigid_Sym_SS = Get_State_Space_Rigid_Wing(Stru, Aero, N, geo, aero_option)
eig_rigid_us = np.linalg.eigvals(Rigid_Sym_SS["A_rigid"])
plt.plot(eig_rigid_us.real, eig_rigid_us.imag, "b*", label="unsteady aerodynamics")

# Quasi-steady aerodynamics
aero_option = 1
Rigid_Sym_SS = Get_State_Space_Rigid_Wing(Stru, Aero, N, geo, aero_option)
eig_rigid_qs = np.linalg.eigvals(Rigid_Sym_SS["A_rigid"])
plt.plot(eig_rigid_qs.real, eig_rigid_qs.imag, "ro", label="quasi-steady aerodynamics")

plt.axvline(x=0, color="k", linestyle="--", linewidth=0.8)
plt.grid(True)
plt.legend()
plt.xlabel("Real part [1/s]")
plt.ylabel("Imaginary part [rad/s]")
plt.title("Eigenvalues of a rigid clamped wing beam")


# =========================================================================
# Zoom near origin for both cases
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Rigid wing - zoom
axes[0].plot(eig_rigid_qs.real, eig_rigid_qs.imag, "ro", label="quasi-steady")
axes[0].plot(eig_rigid_us.real, eig_rigid_us.imag, "b*", label="unsteady")
axes[0].axvline(x=0, color="k", linestyle="--", linewidth=0.8)
axes[0].set_xlim([-20, 5])
axes[0].set_ylim([-50, 50])
axes[0].set_xlabel("Real part [1/s]")
axes[0].set_ylabel("Imaginary part [rad/s]")
axes[0].set_title("Rigid - zoom near origin")
axes[0].legend()
axes[0].grid(True)

# Flexible wing - zoom
axes[1].plot(eig_flex_qs.real, eig_flex_qs.imag, "ro", label="quasi-steady")
axes[1].plot(eig_flex_us.real, eig_flex_us.imag, "b*", label="unsteady")
axes[1].axvline(x=0, color="k", linestyle="--", linewidth=0.8)
axes[1].set_xlim([-20, 5])
axes[1].set_ylim([-50, 50])
axes[1].set_xlabel("Real part [1/s]")
axes[1].set_ylabel("Imaginary part [rad/s]")
axes[1].set_title("Flexible - zoom near origin")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()


# -------------------------------------------------------------------------
# Print basic stability information
# -------------------------------------------------------------------------
print("\n===== Stability check =====")
print("Rigid QS max real part:     ", np.max(eig_rigid_qs.real))
print("Rigid US max real part:     ", np.max(eig_rigid_us.real))
print("Flexible QS max real part:  ", np.max(eig_flex_qs.real))
print("Flexible US max real part:  ", np.max(eig_flex_us.real))

print("\nNumber of eigenvalues:")
print("Rigid QS:    ", len(eig_rigid_qs))
print("Rigid US:    ", len(eig_rigid_us))
print("Flexible QS: ", len(eig_flex_qs))
print("Flexible US: ", len(eig_flex_us))

plt.show()