import numpy as np

def Get_2D_Aerodynamics(geo, flow):
    # Get the aerodynamic matrices for a 2D typical wing section

    # get input parameters
    b = geo['b']
    a = geo['a']
    c = geo['c']

    rho = flow['rho']
    Vel = flow['Vel']

    # Wagner function parameters
    A1 = 0.2048
    A2 = 0.2952
    b1 = 0.0557
    b2 = 0.3330

    # Kussner function parameters
    A3 = 0.5792
    A4 = 0.4208
    b3 = 0.1393
    b4 = 1.802

    # Theodorsen parameters
    T1  = -1/3*np.sqrt(1-c**2)*(2+c**2)+c*np.arccos(c)
    T2  = c*(1-c**2)-np.sqrt(1-c**2)*(1+c**2)*np.arccos(c)+c*(np.arccos(c))**2
    T3  = -(1/8+c**2)*(np.arccos(c))**2+1/4*c*np.sqrt(1-c**2)*np.arccos(c)*(7+2*c**2)-1/8*(1-c**2)*(5*c**2+4)
    T4  = -np.arccos(c)+c*np.sqrt(1-c**2)
    T5  = -(1-c**2)-(np.arccos(c))**2+2*c*np.sqrt(1-c**2)*np.arccos(c)
    T6  = T2
    T7  = -(1/8+c**2)*np.arccos(c)+1/8*c*np.sqrt(1-c**2)*(7+2*c**2)
    T8  = -1/3*np.sqrt(1-c**2)*(2*c**2+1)+c*np.arccos(c)
    T9  = 1/2*(1/3*(1-c**2)**(3/2)+a*T4)
    T10 = np.sqrt(1-c**2)+np.arccos(c)
    T11 = np.arccos(c)*(1-2*c)+np.sqrt(1-c**2)*(2-c)
    T12 = np.sqrt(1-c**2)*(2+c)-np.arccos(c)*(2*c+1)
    T13 = 1/2*(-T7-(c-a)*T1)
    T14 = 1/16+0.5*a*c

    # Noncirculatory lift
    Ma_non = -rho*b**2*np.array([
        [np.pi, -b*np.pi*a, -T1*b],
        [-b*a*np.pi, b**2*np.pi*(1/8+a**2), -(T7+(c-a)*T1)*b**2],
        [-b*T1, 2*b**2*T13, -b**2*T3/np.pi]
    ])

    Ca_non = -rho*b**2*Vel*np.array([
    [0, np.pi, -T4],
    [0, np.pi*(0.5-a)*b, (T1-T8-(c-a)*T4+T11/2)*b],
    [0, (-2*T9-T1+T4*(a-1/2))*b, -b*T4*T11/(2*np.pi)]
    ])

    Ka_non = -rho*b**2*Vel**2*np.array([
        [0,  0,  0],
        [0,  0,  T4+T10],
        [0,  0,  (T5-T4*T10)/np.pi]
    ])

    # Circulatory lift
    # half of the lift build up instantly
    Ca_cir = (1-A1-A2)*(2*np.pi*rho*b*Vel)*np.array([
        [-1], [b*(a+0.5)], [-T12*b/2/np.pi]
    ]).dot(np.array([
        [1,  b*(0.5-a), T11*b/2/np.pi]
    ]))

    Ka_cir = (1-A1-A2)*(2*np.pi*rho*b*Vel)*np.array([
        [-1], [b*(a+0.5)], [-T12*b/2/np.pi]
    ]).dot(np.array([
        [0, Vel, T10*Vel/np.pi]
    ]))

    # lagged aerodynamics, input --> alpha_3/4
    K_lag = 2*np.pi*rho*b*Vel**2*np.array([
        [-1], [b*(a+0.5)], [-T12*b/2/np.pi]
    ]).dot(np.array([
        [(A1+A2)*b1*b2*(Vel/b)**2, (A1*b1+A2*b2)*(Vel/b), (A3+A4)*b3*b4*(Vel/b)**2, (A3*b3+A4*b4)*(Vel/b)]
    ]))

    # Lag state dynamics (dof --> x1,x2; gusts --> x3,x4)
    A_lag = np.array([
        [0, 1, 0, 0],
        [-b1*b2*(Vel/b)**2, -(b1+b2)*Vel/b, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -b3*b4*(Vel/b)**2, -(b3+b4)*Vel/b]
    ])

    AC_lag = np.zeros((4,3))
    AC_lag[1,:] = np.array([1, b * (0.5 - a), T11 * b / (2 * np.pi)])/ Vel

    AK_lag = np.zeros((4,3))
    AK_lag[1,:] = np.array([0, Vel, T10*Vel/np.pi])/Vel

    # Direct influence of rigid body AOA --> half of the lift build up directly
    K_AOA_r = (1-A1-A2)*(2*np.pi*rho*b*Vel**2)*np.array([
        [-1], [b*(a+0.5)], [-T12*b/2/np.pi]
    ])

    # Outputs
    Aero = {
        'Ma_non': Ma_non,
        'Ca_non': Ca_non,
        'Ka_non': Ka_non,
        'Ca_cir': Ca_cir,
        'Ka_cir': Ka_cir,
        'K_lag': K_lag,
        'A_lag': A_lag,
        'AC_lag': AC_lag,
        'AK_lag': AK_lag,
        'K_AOA_r': K_AOA_r
    }
    
    return Aero



