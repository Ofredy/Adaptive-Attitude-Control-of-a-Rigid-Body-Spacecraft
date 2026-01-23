import numpy as np


def get_tilde_matrix(vec):

    v1, v2, v3 = vec[0], vec[1], vec[2]

    return np.array([[ 0, -v3, v2 ],
                     [ v3, 0, -v1 ],
                     [ -v2, v1, 0 ]])