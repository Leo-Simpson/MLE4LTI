import casadi as ca
import numpy as np
import contextlib
import sys
from .misc import Riccati, create_symbolic_triangular, create_symbolic_rectangular, revert_param_triangular
from .sequantialprogramming import SP
from .interiorpoint import construct_nlp


def solve(problem, theta0, formulation, algorithm, opts=None, verbose=False,
    min_eigP = 1e-6,
    min_eigS = 1e-6
):
    if opts is None:
        opts = {}
    assert formulation in ["PredErr", "MLE"], \
        f"Formulation {formulation} is unknown. Choose between 'PredErr' or 'MLE'"
    assert algorithm in ["SP", "IPOPT-lifted", "IPOPT-dense"], \
        f"Algorithm {algorithm} is unknown."\
        "Choose between 'SP' or 'IPOPT-lifted' or 'IPOPT-dense' "
    theta = None
    if verbose:
        stdout = sys.stdout
    else:
        stdout = open('nul', 'w', encoding='utf-8')
    with contextlib.redirect_stdout(stdout):
        if algorithm == "SP":
            lifted = None
            optimizer_constructer = SP
        elif algorithm[:5] == "IPOPT":
            lifted = algorithm == "IPOPT-lifted"
            optimizer_constructer = construct_nlp
        pb_simplier = reformulate(problem, riccati=True, min_eigP=min_eigP, min_eigS=min_eigS)
        ny = pb_simplier["C"].shape[0]
        G_fn_fullS = construct_G(ny, formulation)
        
        V_sym = ca.SX.sym("V", ny, ny)
        G_sym = G_fn_fullS(V_sym, pb_simplier["chol_S"])
        G_fn = ca.Function("G", [V_sym, pb_simplier["eta_S"]], [G_sym])

        optimizer = optimizer_constructer(
            pb_simplier["us"], pb_simplier["x0"], theta0,
            pb_simplier["theta"], pb_simplier["eta_L"], pb_simplier["eta_P"], pb_simplier["eta_S"],
            pb_simplier["A"], pb_simplier["B"], pb_simplier["C"], pb_simplier["D"],
            pb_simplier["equality_constraint"], pb_simplier["inequality_constraint"],
            pb_simplier["build_ACQR"], pb_simplier["build_eta"],
            G_fn, opts, lifted,
            names_eq=pb_simplier["names_eq"], names_ineq=pb_simplier["names_ineq"]
        )
        theta, stats = optimizer.solve()
    return theta, stats

def reformulate(problem,
    riccati=True,
    min_eigP = 1e-6,
    min_eigS = 1e-6
):
    """
        reformulate the problem into the following optimization problem:
            minimize  G( S(eta_S),   1/N sum e_k e_k^T ) 
                    s.t.
                        x_{k+1} = A(theta, eta_L) x_k + B(theta, eta_L) u_k
                        e_k = C(theta) x_k + D(theta) u_k
                        equality_constraint(theta, eta_P, eta_S, eta_L) = 0
                        inequality_constraint(theta) > 0
        where the optimization variables are theta, eta_P, eta_S, eta_L and the sequences x_k, e_k.

        The function G depends on which formulation is used.
    """
    model = problem["model"]
    list_ys = problem["ys"]
    list_us = problem["us"]
    list_x0 = problem["x0"]
    if not isinstance(list_ys, list):
        list_ys = [list_ys]
        list_us = [list_us]
    if not isinstance(list_x0, list):
        list_x0 = [list_x0] * len(list_ys)

    # define the inital linear system
    theta_sym = ca.SX.sym("theta", model.ntheta)
    lti = model.get_linearization()(theta=theta_sym)
    A, B, C, D = lti["A"], lti["B"], lti["C"], lti["D"]
    b, d = lti["b"], lti["d"]
    Q, R = lti["Q"], lti["R"]
    build_ACQR = ca.Function("ACQR", [theta_sym], [A, C, Q, R])

    # define P, S, L using Cholesky parameterization
    eta_P_sym, chol_P = create_symbolic_triangular("chol_P", model.nx)
    P = chol_P @ chol_P.T
    eta_S_sym, chol_S = create_symbolic_triangular("chol_S", model.ny)
    S = chol_S @ chol_S.T
    eta_L_sym, L = create_symbolic_rectangular("L", model.nx, model.ny)

    # define equality constraints with respect to Riccati equations
    eq_P = A @ P @ A.T - L @ S @ L.T + Q - P
    eq_S = (C @ P @ C.T + R) - S
    eq_L = A @ P @ C.T - L @ S

    # define equality constraints
    eq = ca.SX()
    names_eq = []
    if riccati:
        eq_P = revert_param_triangular(eq_P)
        eq_S = revert_param_triangular(eq_S)
        eq_L = ca.reshape(eq_L, -1, 1)
        eq = ca.vertcat(eq, eq_P, eq_S, eq_L)
        names_eq =["eq_P"] * eq_P.shape[0] + ["eq_S"] * eq_S.shape[0] + ["eq_L"] * eq_L.shape[0]
    # define inequality constraints
    ineq1 = model.Ineq(theta_sym)
    ineq_P = ca.diag(chol_P) - min_eigP
    ineq_S = ca.diag(chol_S) - min_eigS
    ineq = ca.vertcat(ineq1, ineq_P, ineq_S)
    names_ineq = ["ineq1"] * ineq1.shape[0] + \
        ["ineq_P"] * ineq_P.shape[0] + ["ineq_S"] * ineq_S.shape[0]


    # define a function to build eta
    P_sym = ca.SX.sym("P", model.nx, model.nx)
    chol_P_sym = ca.chol((P_sym + P_sym.T)/2.).T
    eta_P_revert = revert_param_triangular(chol_P_sym)
    S_sym = ca.SX.sym("S", model.ny, model.ny)
    chol_S_sym = ca.chol((S_sym + S_sym.T)/2.).T
    eta_S_revert = revert_param_triangular(chol_S_sym)

    build_etaP = ca.Function("eta", [P_sym], [eta_P_revert])
    build_etaS = ca.Function("eta", [S_sym], [eta_S_revert])
    build_etaL = ca.Function("eta", [L], [eta_L_sym])

    # Stack everything
    problem_simplier = {
        "us": [ np.hstack([u, np.ones((len(u), 1)), y]) for (u, y) in zip(list_us, list_ys) ],
        "x0": list_x0,
        "theta": theta_sym,
        "eta_P": eta_P_sym,
        "eta_S": eta_S_sym,
        "eta_L": eta_L_sym,
        "chol_S": chol_S,
        "A": A - L @ C,
        "B": ca.horzcat(B, b - L @ d, L),
        "C": C,
        "D":  ca.horzcat(D, d, -ca.DM.eye(model.ny)),
        "Q": Q,
        "R": R,
        "equality_constraint": eq,
        "inequality_constraint": ineq,
        "build_ACQR": build_ACQR,
        "build_eta": {"P":build_etaP, "S":build_etaS, "L":build_etaL},
        "names_eq": names_eq,
        "names_ineq": names_ineq
    }
    return problem_simplier

def construct_G(ny, formulation):
    V = ca.SX.sym("V", ny, ny)
    chol_S = ca.SX.sym("chol_S", ny, ny)
    if formulation == "PredErr":
        G = ca.trace(V)
    elif formulation == "MLE":
        invchol_S = ca.inv(chol_S)
        G = ca.trace(invchol_S @ V @ invchol_S.T) + logdet(chol_S)
    scaling_cost = 1e2
    G = scaling_cost * G
    G_fn = ca.Function("G", [V, chol_S], [G])
    return G_fn

def logdet(chol):
    return 2 * ca.sum1(ca.log(ca.diag(chol)))

# For other purpose
def kalman(model, x0, us, ys, theta):
    N = len(us)
    x = x0.copy()
    hat_x = np.empty((N, model.nx))
    hat_y = np.empty((N, model.ny))
    linear_model = model.get_linearization()(theta=theta)
    _, _, L = Riccati(
        linear_model["A"].full(), linear_model["C"].full(),
        linear_model["Q"].full(), linear_model["R"].full())
    for k in range(N):
        hat_y[k, :] = model.G(x, us[k], theta).full().squeeze()
        hat_x[k, :] = x
        x = model.Fdiscr(x, us[k], theta).full().squeeze() + L @ (ys[k] - hat_y[k])
    return hat_x, hat_y

def compute_cost(problem, theta, formulation):
    linear_model = problem["model"].get_linearization()(theta=theta)
    _, S, L = Riccati(
        linear_model["A"].full(), linear_model["C"].full(),
        linear_model["Q"].full(), linear_model["R"].full())
    list_ys = problem["ys"]
    list_us = problem["us"]
    list_x0 = problem["x0"]
    if not isinstance(list_ys, list):
        list_ys = [list_ys]
        list_us = [list_us]
    if not isinstance(list_x0, list):
        list_x0 = [list_x0] * len(list_ys)
    V  = 0.
    Ntot = 0
    for us, ys, x0 in zip(list_us, list_ys, list_x0):
        N = len(us)
        x = x0.copy()
        for k in range(N):
            hat_y = problem["model"].G(x, us[k], theta).full().squeeze()
            e = ys[k] - hat_y
            x = problem["model"].Fdiscr(x, us[k], theta).full().squeeze() + L @ e

            V = V + np.outer(e, e)
            Ntot += 1
    V = V / Ntot

    G_fn = construct_G(S.shape[0], formulation)
    chol_S = np.linalg.cholesky(S)
    return G_fn(V, chol_S).full().squeeze()