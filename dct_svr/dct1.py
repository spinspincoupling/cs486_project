from math import *
from numpy import *

# useful link: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

# x is a numpy matrix, k_max is a positive integer
def dct1(x, k_max=infty):

    N = x.shape[0]
    k_max = min(k_max, N)

    A = cos((pi/N*arange(0.,k_max)[:, newaxis]) @ (arange(0.,N-1)+5))
    # not too sure about this transpose... but it does make the error go away :D
    X = A @ x

    return A, X
