import numpy as np

def Get_FEM_Beam_Structure(geo, stiff, mass, N):
    # GET_2D_STRUCTURE
    # get the mass, damping, and stiffness matrices for a clamped beam
    # each node has four Dof: heave, bending, torsion, flap (w,phi,theta,beta)

    # Degrees of freedom
    Nn = N['Nn']   # number of nodes
    Ne = N['Ne']   # number of elements
    Nv = N['Nv']   # number of \dot q
    Nd = N['Nd']   # number of q (w,phi,theta,beta)

    # Geometric parameters
    b = geo['b']
    a = geo['a']
    c = geo['c']
    xcg = geo['xcg']
    L = geo['L']

    # Stiffness parameters
    E = stiff['E']
    G = stiff['G']
    Ixx = stiff['Ixx']
    J = stiff['J']
    Kf = stiff['Kf']
    psic = stiff['psic']
    Kc = stiff['Kc']

    # Mass parameters
    m = mass['m']
    It = mass['It']
    mf = mass['mf']
    If_cgf = mass['If_cgf']
    xf = mass['xf']
    Sf = mass['Sf']
    If = mass['If']

    # Dof selection
    EFT_beam = np.zeros((Ne, 8), dtype=int)
    for i in range(Ne):
        EFT_beam[i] = (i * 4) + np.array([1, 2, 3, 4, 5, 6, 7, 8]) - 1

    # Beam mass matrix
    Ms = np.zeros((Nv, Nv))

    for i in range(Ne):
        Me = np.zeros((6, 6))

        Mbb = np.array([
            [(13 * L[i] * m[i]) / 35, (11 * L[i] ** 2 * m[i]) / 210, (9 * L[i] * m[i]) / 70, -(13 * L[i] ** 2 * m[i]) / 420],
            [(11 * L[i] ** 2 * m[i]) / 210, (L[i] ** 3 * m[i]) / 105, (13 * L[i] ** 2 * m[i]) / 420, -(L[i] ** 3 * m[i]) / 140],
            [(9 * L[i] * m[i]) / 70, (13 * L[i] ** 2 * m[i]) / 420, (13 * L[i] * m[i]) / 35, -(11 * L[i] ** 2 * m[i]) / 210],
            [-(13 * L[i] ** 2 * m[i]) / 420, -(L[i] ** 3 * m[i]) / 140, -(11 * L[i] ** 2 * m[i]) / 210, (L[i] ** 3 * m[i]) / 105]
        ])
        Mtb = np.array([
            [(7 * L[i] * m[i] * xcg[i]) / 20, (L[i] ** 2 * m[i] * xcg[i]) / 20, (3 * L[i] * m[i] * xcg[i]) / 20, -(L[i] ** 2 * m[i] * xcg[i]) / 30],
            [(3 * L[i] * m[i] * xcg[i]) / 20, (L[i] ** 2 * m[i] * xcg[i]) / 30, (7 * L[i] * m[i] * xcg[i]) / 20, -(L[i] ** 2 * m[i] * xcg[i]) / 20]
        ])
        Mbt = np.array([
            [(7 * L[i] * m[i] * xcg[i]) / 20, (3 * L[i] * m[i] * xcg[i]) / 20],
            [(L[i] ** 2 * m[i] * xcg[i]) / 20, (L[i] ** 2 * m[i] * xcg[i]) / 30],
            [(3 * L[i] * m[i] * xcg[i]) / 20, (7 * L[i] * m[i] * xcg[i]) / 20],
            [-(L[i] ** 2 * m[i] * xcg[i]) / 30, -(L[i] ** 2 * m[i] * xcg[i]) / 20]
        ])
        Mtt = np.array([
            [It[i] * L[i] / 3, It[i] * L[i] / 6],
            [It[i] * L[i] / 6, It[i] * L[i] / 3]
        ])
        Me[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = Mbb
        Me[np.ix_([2, 5], [2, 5])] = Mtt
        Me[np.ix_([2, 5], [0, 1, 3, 4])] = Mtb
        Me[np.ix_([0, 1, 3, 4], [2, 5])] = Mbt

        Ms[np.ix_(EFT_beam[i, [0, 1, 2, 4, 5, 6]], EFT_beam[i, [0, 1, 2, 4, 5, 6]])] += Me

        Ms[EFT_beam[i, 3], EFT_beam[i, 3]] = If[i]  # I_beta
        Ms[EFT_beam[i, 0], EFT_beam[i, 3]] = Sf[i]  # S_beta
        Ms[EFT_beam[i, 2], EFT_beam[i, 3]] = If[i] + Sf[i] * (c[i] - a[i]) * b[i]  # I_beta + b(c-a)S_beta
        Ms[EFT_beam[i, 3], EFT_beam[i, 0]] = Sf[i]
        Ms[EFT_beam[i, 3], EFT_beam[i, 2]] = If[i] + Sf[i] * (c[i] - a[i]) * b[i]
        
    if i == Ne-1:
        Ms[EFT_beam[i, 7], EFT_beam[i, 7]] = If[i + 1]
        Ms[EFT_beam[i, 4], EFT_beam[i, 7]] = Sf[i + 1]
        Ms[EFT_beam[i, 6], EFT_beam[i, 7]] = If[i + 1] + Sf[i + 1] * (c[i + 1]
        - a[i + 1]) * b[i + 1]
        Ms[EFT_beam[i, 7], EFT_beam[i, 4]] = Sf[i + 1]
        Ms[EFT_beam[i, 7], EFT_beam[i, 6]] = If[i + 1] + Sf[i + 1] * (c[i + 1] - a[i + 1]) * b[i + 1]
     
    # Beam stiffness matrix
    Ks = np.zeros((Nd, Nd))

    for i in range(Ne):
        Ke = np.zeros((6, 6))

        Kb = np.array([
            [(12 * E * Ixx[i]) / L[i] ** 3, (6 * E * Ixx[i]) / L[i] ** 2, -(12 * E * Ixx[i]) / L[i] ** 3, (6 * E * Ixx[i]) / L[i] ** 2],
            [(6 * E * Ixx[i]) / L[i] ** 2, (4 * E * Ixx[i]) / L[i], -(6 * E * Ixx[i]) / L[i] ** 2, (2 * E * Ixx[i]) / L[i]],
            [-(12 * E * Ixx[i]) / L[i] ** 3, -(6 * E * Ixx[i]) / L[i] ** 2, (12 * E * Ixx[i]) / L[i] ** 3, -(6 * E * Ixx[i]) / L[i] ** 2],
            [(6 * E * Ixx[i]) / L[i] ** 2, (2 * E * Ixx[i]) / L[i], -(6 * E * Ixx[i]) / L[i] ** 2, (4 * E * Ixx[i]) / L[i]]
        ])
        Kt = np.array([
            [(G * J[i]) / L[i], -(G * J[i]) / L[i]],
            [-(G * J[i]) / L[i], (G * J[i]) / L[i]]
        ])
        Kbt = np.array([
            [0, 0],
            [Kc[i] / 4, -Kc[i] / 4],
            [0, 0],
            [-Kc[i] / 4, Kc[i] / 4]
        ])
        Ktb = np.array([
            [0, Kc[i] / 4, 0, -Kc[i] / 4],
            [0, -Kc[i] / 4, 0, Kc[i] / 4]
        ])
        Ke[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = Kb
        Ke[np.ix_([2, 5], [2, 5])] = Kt
        Ke[np.ix_([0, 1, 3, 4], [2, 5])] = Kbt
        Ke[np.ix_([2, 5], [0, 1, 3, 4])] = Ktb

        Ks[np.ix_(EFT_beam[i, [0, 1, 2, 4, 5, 6]], EFT_beam[i, [0, 1, 2, 4, 5, 6]])] += Ke

        Ks[EFT_beam[i, 3], EFT_beam[i, 3]] = Kf[i]

    if i == Ne-1:
        Ks[EFT_beam[i, 7], EFT_beam[i, 7]] = Kf[i + 1]

    # Damping matrix
    Cs = 1e-6 * Ks

    # Outputs
    Stru = {}
    Stru['Ms'] = Ms
    Stru['Cs'] = Cs
    Stru['Ks'] = Ks

    return Stru
