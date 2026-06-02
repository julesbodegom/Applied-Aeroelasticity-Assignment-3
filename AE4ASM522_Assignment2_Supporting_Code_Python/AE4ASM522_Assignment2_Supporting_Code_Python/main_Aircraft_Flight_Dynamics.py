import numpy as np
import matplotlib.pyplot as plt
import scipy
from Get_Glider_Properties import Get_Glider_Properties 
from Get_Coupling_Matrices import Get_Coupling_Matrices
from Get_Aero_Jacobian_Tail import Get_Aero_Jacobian_Tail
from Get_EQM_Coupled_Flight_Dynamics import Get_EQM_Coupled_Flight_Dynamics
from Get_EQM_Rigid_Dof import Get_EQM_Rigid_Dof
from skew import skew

# Flow parameters
flow={}
flow['rho'] = 1.225
flow['Vel'] = 35

# Glider Overall Properties
par = Get_Glider_Properties()

# Aerodynamic influences from the tail
Jco_Tail = Get_Aero_Jacobian_Tail(par, flow)

# Equations of motion for rigid-body degrees of freedom
Rigid_Dof = Get_EQM_Rigid_Dof(par, flow, Jco_Tail)

# Coupling matrices between rigid-body and aeroelastic Dof
Coup = Get_Coupling_Matrices(par, flow)

# Options
option={}
option['aero'] = 1 # 1-Quasi-steady aerodynamics  0-unsteady aerodynamics 
option['stru'] = 1 # Rigid wing  0-elastic wing 

# Fully coupled flight dynamic equations
Fdyn_Sys_SS = Get_EQM_Coupled_Flight_Dynamics(Rigid_Dof, Coup, flow, par, option)

# Model analysis
Rigid_AC_Quasi_Aero = Fdyn_Sys_SS

eig_rigid = np.linalg.eigvals(Rigid_AC_Quasi_Aero['A_Fdyn'])

option={}
option['aero'] = 0 # 1-Quasi-steady aerodynamics  0-unsteady aerodynamics 
option['stru'] = 0 # Rigid wing  0-elastic wing 
Flexible_AC_Un_Aero = Get_EQM_Coupled_Flight_Dynamics(Rigid_Dof, Coup, flow, par, option)

eig_flex = np.linalg.eigvals(Flexible_AC_Un_Aero['A_Fdyn'])

plt.plot(np.real(eig_flex), np.imag(eig_flex), 'b*')
plt.plot(np.real(eig_rigid), np.imag(eig_rigid), 'ro')
plt.grid(True)
plt.legend(['Flexible AC unsteady aero', 'Rigid AC quasi-steady aero'])
plt.title('Eigenvalues of a free-flying aircraft')
plt.show()

# Save models
# np.savez('AC_Fdyn_Sys_SS', Rigid_AC_Quasi_Aero=Rigid_AC_Quasi_Aero, Flexible_AC_Un_Aero=Flexible_AC_Un_Aero)
