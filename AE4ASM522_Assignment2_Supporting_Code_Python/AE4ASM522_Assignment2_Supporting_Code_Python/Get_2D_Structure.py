import numpy as np

def Get_2D_Structure(geo):
    # Stiffness and damping parameters
    K_h = 2131.8346
    K_theta = 198.9712
    K_beta = 17.3489

    Ks = np.diag([K_h, K_theta, K_beta])
    Cs = Ks / 1000

    # Geometric parameters
    b = geo['b']
    a = geo['a']
    c = geo['c']

    # Mass matrix
    option = 0   # direct parameters
    # option = 1  # use the distributed parameters

    if option == 0:
        m = 13.5
        S_theta = 0.3375
        S_beta = 0.1055
        I_theta =  0.0787
        I_theta_beta = 0.0136
        I_beta = 0.0044

        Ms = np.array([
            [m, S_theta, S_beta],
            [S_theta, I_theta, I_theta_beta],
            [S_beta, I_theta_beta, I_beta]
        ])

    elif option == 1:
        ma = 1.567
        mf = 0.1
        Icg_theta = 1
        Icg_beta = 0.01

        x_theta = 0.1
        x_beta = 0.1

        S_theta = ma*x_theta*b+mf*(c-a+x_beta)*b
        S_beta = mf*x_beta*b
        I_theta = ma*(x_theta*b)**2+mf*(c-a+x_beta)**2*b**2+Icg_theta+Icg_beta
        I_beta = mf*(x_beta*b)**2+Icg_beta

        Ms = np.array([
            [ma+mf, S_theta, S_beta],
            [S_theta, I_theta, I_beta+(c-a)*b*S_beta],
            [S_beta, I_beta+(c-a)*b*S_beta, I_beta]
        ])

    # Outputs
    Stru = {
        'Ms': Ms,
        'Cs': Cs,
        'Ks': Ks
    }

    return Stru
