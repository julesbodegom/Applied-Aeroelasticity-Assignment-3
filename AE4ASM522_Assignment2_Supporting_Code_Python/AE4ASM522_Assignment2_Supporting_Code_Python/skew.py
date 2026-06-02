import numpy as np

def skew(vector):
    matrix = np.array([[0, -vector[2], vector[1]],
                       [vector[2], 0, -vector[0]],
                       [-vector[1], vector[0], 0]])
    return matrix
