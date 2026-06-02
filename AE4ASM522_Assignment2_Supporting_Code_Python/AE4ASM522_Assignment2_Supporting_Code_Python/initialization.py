import numpy as np

# initializations for simulation 

# to be completed by you 

# load the state-space flight dynamic file that you have created
#aero_data = scipy.io.loadmat('AC_Fdyn_Sys_SS.mat')

d2r = np.pi / 180
r2d = 180 / np.pi

# gust inputs
gust_mag = 5  # in degree
gust_freq = 0.2  # in Herz

# rigid aircraft model with quasi steady aerodynamics
# AC_Fdyn_Sys_SS, states definition (total 40 states)
# rigid states(12)|
# right wing: \dot flap states (7) +  flap states (7)
# left  wing:  dot flap states (7) +  flap states (7)

N_rigid = Rigid_AC_Quasi_Aero.A_Fdyn.shape[0]

index_rigid = np.arange(1, 13)
index_right_flap = np.arange(20, 27)
index_left_flap = np.arange(34, 41)

# input definition
# B_Fdyn_delta [delta a, delta_e, delta_r]
# B_Fdyn_gust has 14 inputs (local gust AOA on each strip)
# if assumes uniform gust
H_gust_selection = np.ones(Rigid_AC_Quasi_Aero.B_Fdyn_gust.shape[1])  # uniform gust selection vector

# B_Fdyn_flap has 14 inputs --> flap moment input at each strip
# order right wing root --> right wing tip --> left wing root --> left wing tip
# assume positive is defined as right down (tip two strips) and left up
H_flap_selection = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, -1, -1])

Rigid_B_Fdyn_sorted = np.column_stack((
    Rigid_AC_Quasi_Aero.B_Fdyn_delta,
    Rigid_AC_Quasi_Aero.B_Fdyn_flap @ H_flap_selection,
    Rigid_AC_Quasi_Aero.B_Fdyn_gust @ H_gust_selection
))

# flexible aircraft model with unsteady aerodynamics
# Flexible_AC_Un_Aero, states definition (total 180 states)
# rigid states(12)|
# right wing: \dot x_e (28) + x_e (28)+ z lag (28)
# left  wing: \dot x_e (28) + x_e (28)+ z lag (28)

N_flex = Flexible_AC_Un_Aero.A_Fdyn.shape[0]

index_rw_h = np.arange(41, 69, 4)
index_rw_theta = np.arange(43, 69, 4)
index_rw_beta = np.arange(44, 69, 4)

index_lw_h = index_rw_h + 84
index_lw_theta = index_rw_theta + 84
index_lw_beta = index_rw_beta + 84

# input definition
# B_Fdyn_delta [delta a, delta_e, delta_r]
# B_Fdyn_gust has 14 inputs (local gust AOA on each strip)
# if assumes uniform gust
H_gust_selection = np.ones(Flexible_AC_Un_Aero.B_Fdyn_gust.shape[1])  # uniform gust selection vector

# B_Fdyn_flap has 14 inputs --> flap moment input at each strip
# order right wing root --> right wing tip --> left wing root --> left wing tip
# assume positive is defined as right down (tip two strips) and left up
H_flap_selection = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, -1, -1])

Flex_B_Fdyn_sorted = np.column_stack((
    Flexible_AC_Un_Aero.B_Fdyn_delta,
    Flexible_AC_Un_Aero.B_Fdyn_flap @ H_flap_selection,
    Flexible_AC_Un_Aero.B_Fdyn_gust @ H_gust_selection
))


