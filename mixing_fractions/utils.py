import numpy as np
from numba import njit
from figaro.load import load_data

@njit
def logsumexp_jit(a, b):
    a_max = np.max(a)
    tmp = b * np.exp(a - a_max)
    return np.log(np.sum(tmp)) + a_max
