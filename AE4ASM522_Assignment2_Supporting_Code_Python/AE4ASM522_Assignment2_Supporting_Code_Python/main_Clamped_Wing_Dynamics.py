import numpy as np
import matplotlib.pyplot as plt
import scipy

from Get_Beam_Strip_Aerodynamics import Get_Beam_Strip_Aerodynamics
from Get_State_Space_Elastic_Wing import Get_State_Space_Elastic_Wing
from Get_FEM_Beam_Structure import Get_FEM_Beam_Structure
from Get_State_Space_Rigid_Wing import Get_State_Space_Rigid_Wing

# Constants
r2d = 180 / np.pi
d2r = np.pi / 180

# Define N as a dictionary
N = {}
N["Nn"] = 8  # number of nodes
N["Ne"] = N["Nn"] - 1  # number of elements 
N["Nv"] = 4 * N["Nn"]  # number of \dot q 
N["Nd"] = 4 * N["Nn"]  # number of q (w,phi,theta,beta)
N["Nz"] = 4 * N["Nn"]  # number for lag states (two for Dof, two for gusts)
N["Ndof"] = N["Nv"] + N["Nd"] + N["Nz"]

N["fxdof"] = [1, 2, 3, 4] + list(np.arange(N["Nv"]+1, N["Nv"]+5)) + list(np.arange(N["Nv"]+N["Nd"]+1, N["Nv"]+N["Nd"]+5))
N["frdof"] = list(set(np.arange(1, N["Ndof"]+1)) - set(N["fxdof"]))

# Define geo as a dictionary
geo = {}
geo["b"] = 0.6 * np.ones(N["Nn"]) / 2  # half chord
geo["a"] = np.zeros(N["Nn"])  # Location of the shear centre wrt midchord
geo["c"] = 0.5 * np.ones(N["Nn"])  # Location of the flap hinge wrt midchord
geo["xcg"] = 0.1 * geo["b"]  # Location of the cg wrt shear centre
geo["Ltot"] = 8  # half span
geo["L"] = geo["Ltot"] / N["Ne"] * np.ones(N["Nn"])  # length per element
geo["Lf"] = (1 - geo["c"]) * geo["b"]

# Define mass as a dictionary
mass = {}
mass["m"] = 0.75 * np.ones(N["Nn"])  # Wing mass per unit length
mass["It"] = 0.1 * np.ones(N["Nn"])
mass["mf"] = 0.25 * np.ones(N["Nn"])  # Flap mass per flap
mass["If_cgf"] = 1e-3 * np.ones(N["Nn"])  # Inertia of the flap around its own cg, per flap  
mass["xf"] = -0.1 * np.ones(N["Nn"])  # Location of the flap cg with respect to the hinge line
mass["Sf"] = mass["mf"] * mass["xf"] * geo["Lf"]
mass["If"] = mass["If_cgf"] + mass["Sf"] * mass["xf"] * geo["Lf"]

# Define stiff as a dictionary
stiff = {}
stiff["E"] = 70e9
stiff["G"] = stiff["E"] / 2 / (1 + 0.3)
stiff["Ixx"] = 1.2e5 / stiff["E"] * np.ones(N["Nn"])  
stiff["J"] = 1e7 / stiff["G"] * np.ones(N["Nn"])  
stiff["Kf"] = 1e2 * np.ones(N["Nn"])    
stiff["psic"] = np.zeros(N["Nn"])  # Bending torsion coupling term
stiff["Kc"] = np.sign(stiff["psic"]) * (stiff["psic"]**2 * 16. * stiff["E"] * stiff["Ixx"] * stiff["G"] * stiff["J"] / geo["L"]**2)**0.5

# Flow parameters
flow = {"rho": 1.225, "Vel": 35}

# Get clamped beam mass, damping, stiffness matrices
Stru = Get_FEM_Beam_Structure(geo, stiff, mass, N)

# Get aerodynamic matrices 
Aero = Get_Beam_Strip_Aerodynamics(geo, flow, N)

# Flexible Beam Assemble to state-space 
# aero_option = 0;   # unsteady aerodynamics
# aero_option = 1; # quasi-steady aerodynamics

# Ae_Sym_SS = Get_State_Space_Elastic_Wing(Stru, Aero, N, geo, aero_option);

# Rigid Beam Assemble to state-space 
# aero_option = 0;   # unsteady aerodynamics
# aero_option = 1; # quasi-steady aerodynamics

# Rigid_Sym_SS = Get_State_Space_Rigid_Wing(Stru, Aero, N, geo, aero_option);

# Flexible clamped wing beam eigenvalue analysis
plt.figure(1)
aero_option = 0
Ae_Sym_SS = Get_State_Space_Elastic_Wing(Stru, Aero, N, geo, aero_option)
eig_un = np.linalg.eig(Ae_Sym_SS["A_ae"])[0]
plt.plot(eig_un.real, eig_un.imag, 'b*')
aero_option = 1
Ae_Sym_SS = Get_State_Space_Elastic_Wing(Stru, Aero, N, geo, aero_option)
eig_quasi = np.linalg.eig(Ae_Sym_SS["A_ae"])[0]
plt.plot(eig_quasi.real, eig_quasi.imag, 'ro')
plt.grid(True) 
plt.legend(['unsteady aerodynamics', 'quasi-steady aerodynamics'])
plt.title('eigenvalues of a flexible clamped wing beam')
# plt.savefig('./figure/wangxuerui.png')

# Rigid clamped wing beam eigenvalue analysis
plt.figure(2) 
aero_option = 0
Ae_Sym_SS = Get_State_Space_Rigid_Wing(Stru, Aero, N, geo, aero_option)
eig_un = np.linalg.eig(Ae_Sym_SS["A_rigid"])[0]
plt.plot(eig_un.real, eig_un.imag, 'b*')
aero_option = 1
Ae_Sym_SS = Get_State_Space_Rigid_Wing(Stru, Aero, N, geo, aero_option)
eig_quasi = np.linalg.eig(Ae_Sym_SS["A_rigid"])[0]
plt.plot(eig_quasi.real, eig_quasi.imag, 'ro')
plt.grid(True)
plt.legend(['unsteady aerodynamics', 'quasi-steady aerodynamics'])
plt.title('eigenvalues of a rigid clamped wing beam')

plt.show()