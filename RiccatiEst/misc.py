import casadi as ca
import numpy as np
import numpy.linalg as LA
from scipy.linalg import solve_discrete_are
from math import ceil

# utils
def create_symbolic_triangular(name, n):
    L = ca.SX.zeros(n, n)
    eta_list = []
    for i in range(n):
        for j in range(i+1):
            etai = ca.SX.sym(f"{name}_{i}_{j}")
            eta_list.append(etai)
            L[i, j] = etai
    eta_vect = ca.vcat(eta_list)
    return eta_vect, L

def revert_param_triangular(L):
    n = L.shape[0]
    eta_list = []
    for i in range(n):
        for j in range(i+1):
            etai = L[i, j]
            eta_list.append(etai)
    eta_vect = ca.vcat(eta_list)
    return eta_vect

def create_symbolic_rectangular(name, n, m):
    eta  = ca.SX.sym(name, n*m)
    return eta, ca.reshape(eta, n, m)


def Riccati(A, C, Q, R):
    P = solve_discrete_are(A.T, C.T, Q, R)
    S = R + C @ P @ C.T
    L = A @ P @ C.T @ LA.inv(S)
    return P, S, L

def generate_u(rng, nu, N, umax=1., umin=0., step=None, step_len=None):
    assert not (step is None and step_len is None), "has to specify one of these"
    if step is None:
        step = ceil(N / step_len)
    if step_len is None:
        step_len = ceil(N / step)
    du = umax - umin
    Us = sum(
        [
        [umin + rng.random(nu) * du] * step_len
        for i in range(step)
        ], [])
    assert len(Us) >= N, "there should be more U than N"
    Us = Us[:N]
    us = np.array(Us)
    return us
