import casadi as ca
import numpy as np
from .nlp import NLP
from .misc import Riccati

def compute_eta(theta, build_ACQR, build_eta):
    A, C, Q, R = build_ACQR(theta) # this is the ACQR of the original problem
    P, S, L = Riccati(A, C, Q, R)
    eta_P = build_eta["P"](P)
    eta_L = build_eta["L"](L)
    eta_S  = build_eta["S"](S)
    return eta_P, eta_L, eta_S

def construct_nlp(list_us, x0s, theta0,
                  theta, eta_L, eta_P, eta_S,
                  A, B, C, D,
                  eq_constraints, ineq_constraints,
                  build_ACQR, build_eta,
                  G_fn, opts, lifted,
                  names_eq=None, names_ineq=None
                  ):

    # initialize with a feasible point
    eta_P0, eta_L0, eta_S0 = compute_eta(theta0, build_ACQR, build_eta)

    nlp = NLP()
    nlp.add_var(theta, theta0)
    nlp.add_var(eta_L, eta_L0)
    nlp.add_var(eta_P, eta_P0)
    nlp.add_var(eta_S, eta_S0)

    nlp.add_eq(eq_constraints, names=names_eq)
    nlp.add_eq(ineq_constraints, ubg=np.inf, names=names_ineq)

    if lifted:
        A_ws = ca.Function("A_ws", [theta, eta_L], [A])(theta0, eta_L0)
        B_ws = ca.Function("B_ws", [theta, eta_L], [B])(theta0, eta_L0)
    V = 0.
    Ntotal = 0
    for i in range(len(list_us)):
        us = list_us[i]
        x0 = x0s[i]
        x = x0.copy()
        x_ws = x0.copy()
        for k in range(us.shape[0]):
            e = C @ x + D @ us[k]
            xplus = A @ x + B @ us[k]
            V = V + e @ e.T
            Ntotal += 1
            if lifted:
                x_ws = A_ws @ x_ws + B_ws @ us[k]
                x = ca.SX.sym(f"x{k+1}", x.shape[0])
                nlp.add_var(x, x_ws)
                nlp.add_eq(x - xplus)
            else:
                x = xplus
    V = V / Ntotal
    cost = G_fn(V, eta_S)
    nlp.add_value(cost)
    nlp.define_important_variable(theta)
    nlp.opts = opts

    nlp.stack()
    # # For debugging, it might be good to check if the NLP is feasible
    # # at the initial point, as it is supposed to.
    # nlp.verif_init(tol=1e-6)
    return nlp
