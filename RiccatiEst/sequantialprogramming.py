import casadi as ca
import numpy as np
import contextlib
from .misc import Riccati

default_opts = {
        "TR_init":1.,
        "TR_shrink":0.5,
        "maxiter":100,
        "rtol.cost_decrease":1e-5,
        "hessian_perturbation":0., 
        "verbose":True
          }

class SP:
    def __init__(self, us, x0s, theta0,
                 theta, eta_L, eta_P, eta_S,
                 A, B, C, D,
                 eq_constraints , ineq_constraints,
                 build_ACQR, build_eta,
                 G_fn, opts, lifted,
                 names_eq=None, names_ineq=None):
        _ = lifted # not used
        if opts is None:
            opts = {}

        self.dims = {
            "x":A.shape[0],
            "y":C.shape[0],
            "u":B.shape[1],
            "theta":theta.shape[0],
            "eta_L":eta_L.shape[0],
            "eta_P":eta_P.shape[0],
            "eta_S":eta_S.shape[0]
        }
        self.x0s = x0s
        self.us = us
        self.list_N = [len(u) for u in us]
        self.N_data = len(us)
        self.theta0 = theta0
        self.opts = self.complete_opts(opts)
        z = ca.vertcat(theta, eta_L)
        self.ABCD_fn = ca.Function("ABCDQR", [z], [A, B, C, D])
        self.build_ACQR_original = build_ACQR
        self.build_eta = build_eta
        self.G_fn = G_fn

        # construct derivatives
        AB_sym = ca.horzcat(A, B)
        CD_sym = ca.horzcat(C, D)
        self.dAB_fn = [
            ca.Function("dAB", [z], [ca.jacobian(AB_sym[i,:], z)])
            for i in range(self.dims["x"])
        ]
        self.dCD_fn = [
            ca.Function("dCD", [z], [ca.jacobian(CD_sym[i,:], z)])
            for i in range(self.dims["y"])
        ]

        # Construct NLP
        self.sub_nlp_long_dict = self.prepare_nlp(
            theta, eta_L, eta_P, eta_S,
            eq_constraints, ineq_constraints, names_eq, names_ineq,
            G_fn
            )

        self.len_TR = 1 * self.opts["TR_init"]

    def complete_opts(self, opts):
        options = default_opts.copy()
        for key, item in opts.items():
            options[key] = item
        return options

    def prepare_nlp(self, theta, eta_L, eta_P, eta_S,
            eq_constraints, ineq_constraints, names_eq, names_ineq,
            G_fn):
        nlpsolver_options = {
            "expand": False,
            "ipopt.max_iter": 500,
            "ipopt.max_cpu_time": 3600.0,
            "ipopt.linear_solver": "mumps",  # suggested: ma57
            "ipopt.mumps_mem_percent": 10000,
            "ipopt.mumps_pivtol": 0.001,
            "ipopt.print_level": 5,
            "ipopt.print_frequency_iter": 10
        }
        
        # Construct V_sym
        theta_bar = ca.SX.sym("theta_bar", self.dims["theta"])
        eta_L_bar = ca.SX.sym("eta_L_bar", self.dims["eta_L"])
        dtheta = theta - theta_bar
        deta_L = eta_L - eta_L_bar

        zbar = ca.vertcat(theta_bar, eta_L_bar)
        nz = self.dims["theta"] + self.dims["eta_L"]
        Vbar = ca.SX.sym("Vbar", self.dims["y"], self.dims["y"])
        V_Jac = {i : ca.SX.sym(f"V_jac_{i}", self.dims["y"], self.dims["y"]) for i in range(nz)}
        V_Hessian = {
            (i,j) : ca.SX.sym(f"V_Hessian_{(i,j)}", self.dims["y"], self.dims["y"])
            for i in range(nz) for j in range(nz)}

        list_p = build_list_variables(zbar, Vbar, V_Jac, V_Hessian)
        p = ca.vcat([ca.reshape(mat, -1, 1) for mat in list_p])

        dz = ca.vertcat(dtheta, deta_L)
        V_sym = 1 * Vbar
        for i in range(nz):
            V_sym = V_sym + dz[i] * V_Jac[i]
            for j in range(nz):
                V_sym = V_sym + 0.5 * dz[i] * dz[j] * V_Hessian[(i,j)]
        cost = G_fn(V_sym, eta_S)

        # add L2 penalty
        cost = cost + self.opts["hessian_perturbation"] * ca.dot(dz, dz)

        # Construct Trust region constraint

        # Construct NLP
        variables = ca.vertcat(theta, eta_L, eta_P, eta_S)
        neq, nineq = eq_constraints.shape[0], ineq_constraints.shape[0]
        g = ca.vertcat(eq_constraints, ineq_constraints, dtheta)

        # Define bounds
        size_TR = ca.SX.sym("size_TR", 1)
        lbg = ca.vertcat(ca.SX.zeros(neq) , ca.SX.zeros(nineq), -size_TR*ca.SX.ones(dtheta.shape[0]) )
        ubg = ca.vertcat(ca.SX.zeros(neq) , np.inf*ca.SX.ones(nineq), size_TR*ca.SX.ones(dtheta.shape[0]))
        lbg_fn = ca.Function("lbg", [size_TR], [lbg])
        ubg_fn = ca.Function("ubg", [size_TR], [ubg])


        nlp = {
            "x":variables,
            "p":p,
            "f":cost,
            "g":g,
        }

        nlp_long_dict = {
            "nlp":nlp,
            "nlpsol": ca.nlpsol("S", "ipopt", nlp, nlpsolver_options),
            "retrieve": ca.Function("retrieve", [variables], [theta]),
            "build_p": ca.Function("build_p", list_p, [p]),
            "build_x": ca.Function("build_x", [theta, eta_P, eta_S, eta_L], [variables],
                                         ["theta", "eta_P", "eta_S", "eta_L"], ["x"]),
            "lbg": lbg_fn,
            "ubg": ubg_fn,
            "g_labels": names_eq + names_ineq + ["trust region"] * dtheta.shape[0]
        }
        return nlp_long_dict

    def dABCD(self, z):
        list_dAB = [dAB(z).full() for dAB in self.dAB_fn]
        list_dCD = [dCD(z).full() for dCD in self.dCD_fn]
        dAB = np.array(list_dAB)
        dCD = np.array(list_dCD)

        dAB_shape = (self.dims["x"],
                     self.dims["x"] + self.dims["u"],
                     self.dims["theta"] + self.dims["eta_L"])
        assert dAB.shape == dAB_shape, f"dAB shape is wrong, got {dAB.shape}, expected {dAB_shape}"

        return dAB, dCD

    def derivatives(self, z, list_xs):
        nz = z.shape[0]
        A, _, C, _ = self.ABCD_fn(z)
        A, C = A.full(), C.full()
        dAB, dCD = self.dABCD(z)
        list_des = []
        for i in range(self.N_data):
            xus = np.hstack([list_xs[i], self.us[i]])
            ws =  myprod(dAB, xus)
            dx0 = np.zeros((self.dims["x"], nz))
            dxs = propagate_dynamics_simple(A, dx0, ws)
            vs = myprod(dCD, xus)
            des = myprod2(C, dxs) + vs
            list_des.append(des)
        return list_des

    def get_derivatives_V(self, list_es, list_des):
        total_value, total_Jac, total_Hessian, total_N = 0., 0., 0., 0.
        for es, des in zip(list_es, list_des):
            value = es.T @ es
            Jac1 = np.einsum("ki,kjt->ijt", es, des)
            Hessian1 = np.einsum("kit,kjs->ijts", des, des)

            Jac = Jac1 + np.swapaxes(Jac1, 0, 1)
            Hessian = Hessian1 + np.swapaxes(Hessian1, 0, 1)

            total_value = total_value + value
            total_Jac = total_Jac + Jac
            total_Hessian = total_Hessian + Hessian
            total_N = total_N + len(es)

        total_value = total_value / total_N
        total_Jac = total_Jac / total_N
        total_Hessian = total_Hessian / total_N
        ntheta_tilde = self.dims["theta"] + self.dims["eta_L"]
        dict_Jac = {}
        dict_Hessian = {}
        for i in range(ntheta_tilde):
            dict_Jac[i] = total_Jac[:,:,i]
            for j in range(ntheta_tilde):
                dict_Hessian[(i,j)] = total_Hessian[:,:,i,j]
        return total_value, dict_Jac, dict_Hessian

    def evaluate(self, theta):
        eta_P, eta_S, eta_L = compute_eta(theta, self.build_ACQR_original, self.build_eta)
        z = np.concatenate([theta, eta_L])
        A, B, C, D = self.ABCD_fn(z)
        A, B, C, D = A.full(), B.full(), C.full(), D.full()
        list_xs, list_es = [], []
        total_V = 0.
        total_N = 0
        for i in range(self.N_data):
            xs, es = propagate_dynamics(A, B, C, D, self.x0s[i], self.us[i])
            V = es.T @ es
            N = len(es)
            total_V = total_V + V
            total_N = total_N + N
            list_xs.append(xs)
            list_es.append(es)
        auxs = {"xs":list_xs, "es":list_es, "eta_L":eta_L, "eta_P":eta_P, "eta_S":eta_S}
        total_V = total_V / total_N
        total_cost = self.G_fn(total_V, eta_S).full().squeeze()
        return auxs, total_cost

    def solve_small_NLP(self, theta, eta_P, eta_S, eta_L, V_value, V_jac, V_hessian):
        x0 = self.sub_nlp_long_dict["build_x"](
                theta=theta, eta_P=eta_P, eta_S=eta_S, eta_L=eta_L)["x"]
        z = np.concatenate([theta, eta_L])
        list_p = build_list_variables(z, V_value, V_jac, V_hessian)
        p = self.sub_nlp_long_dict["build_p"](*list_p)
        # # For debugging, it might be good to check if the NLP is feasible
        # # at the initial point, as it is supposed to.
        # verif_nlp(
        #     self.sub_nlp_long_dict["nlp"],
        #     self.sub_nlp_long_dict["lbg"], self.sub_nlp_long_dict["ubg"],
        #     x0, p, tol=1e-5,
        #     g_labels=self.sub_nlp_long_dict["g_labels"])
        stdout = open('nul', 'w', encoding='utf-8')  # or sys.stdout
        with contextlib.redirect_stdout(stdout):
            solution = self.sub_nlp_long_dict["nlpsol"](
                    x0=x0,
                    p=p,
                    lbg=self.sub_nlp_long_dict["lbg"](self.len_TR).full().squeeze(),
                    ubg=self.sub_nlp_long_dict["ubg"](self.len_TR).full().squeeze(),
            )
        theta = self.sub_nlp_long_dict["retrieve"](solution["x"]).full().squeeze()
        stats = self.sub_nlp_long_dict["nlpsol"].stats()

        # print("theta", theta)
        assert stats["return_status"] == "Solve_Succeeded", \
            f"return_status = {stats['return_status']}"

        return theta

    def stepSQP(self, theta, prev_cost, auxs):
        eta_L = auxs["eta_L"]
        z = np.concatenate([theta, eta_L])
        list_des = self.derivatives(z, auxs["xs"])
        V_value, V_jac, V_hessian = self.get_derivatives_V(auxs["es"], list_des)
        new_theta = self.solve_small_NLP(
            theta, auxs["eta_P"], auxs["eta_S"], eta_L,
            V_value, V_jac, V_hessian)
        new_auxs, new_cost = self.evaluate(new_theta)
        termination = None
        min_dcost = max(abs(prev_cost),abs(new_cost)) * self.opts["rtol.cost_decrease"]
        dcost =  prev_cost - new_cost
        if dcost < 0:
            # The step is rejected
            new_cost = prev_cost
            new_auxs = auxs
            new_theta = theta
            # Needs to shrink the trust region
            self.len_TR = self.len_TR * self.opts["TR_shrink"]
            if self.opts["verbose"]:
                print(f"Shrinking TR to {self.len_TR:.2e}")
            if self.len_TR < 1e-6:
                termination = "TR too small"
        elif dcost < min_dcost:
            # The step is accepted, but small enough to stop
            termination = f"rtol.cost_decrease, cost decrease = {dcost:.2e}"
        else: # dcost > min_dcost
            # The step is accepted
            self.len_TR = self.len_TR / self.opts["TR_shrink"] # Trust Region size increase
            self.len_TR = min(self.len_TR, self.opts["TR_init"]) # Trust Region size not exploding
        return new_theta, new_cost, new_auxs, termination

    def solve(self):
        theta = self.theta0.copy()
        auxs, cost = self.evaluate(theta)
        for j in range(self.opts["maxiter"]):
            theta, cost, auxs, termination  = self.stepSQP(theta, cost, auxs)
            if self.opts["verbose"]:
                print(f"Iteration {j+1}, Cost {cost:2e}, len TR {self.len_TR:.1e}")
            if termination is not None:
                break
        if termination is None:
            termination = "maxiter"
        if self.opts["verbose"]:
            print("termination :", termination)
        stats =  {"return_status":termination,
                  "termination":termination,
                  "niter":j+1,
                  "cost":cost}
        return theta, stats

# utils
def myprod(A, B):
    # Perform the einsum ijz,tj->tiz
    c = np.tensordot(A, B, axes=(1, 1)) # tzi
    return np.moveaxis(c, -1, 0)

def myprod2(A, dxs):
    # Perform the einsum ij,tjz->tiz
    c = np.tensordot(A, dxs, axes=(1, 1)) # tiz
    return np.swapaxes(c, 0, 1)

def propagate_dynamics(A, B, C, D, x0, us):
    """
    Propagate the dynamics of the system
        x_{k+1} = A x_k + B u_k
        y_k = C x_k + D u_k
    """
    Bus = us @ B.T
    xs = propagate_dynamics_simple(A, x0, Bus)
    ys = xs @ C.T + us @ D.T
    return xs, ys

def propagate_dynamics_simple(A, x0, vs):
    """
        Simulate the following:
            x_{k+1} = A x_k + v_k
    """
    N = len(vs)
    xs = np.zeros((N, *x0.shape))
    xs[0] = x0
    for k in range(N-1):
        xs[k+1] = A @ xs[k] + vs[k]
    return xs

def compute_eta(theta, build_ACQR, build_eta):
    A, C, Q, R = build_ACQR(theta) # this is the ACQR of the original problem
    A, C, Q, R = A.full(), C.full(), Q.full(), R.full()
    P, S, L = Riccati(A, C, Q, R)
    eta_P = build_eta["P"](P).full().squeeze()
    eta_S  = build_eta["S"](S).full().squeeze()
    eta_L = build_eta["L"](L).full().squeeze()
    return eta_P, eta_S,  eta_L

def build_list_variables(z, Vbar, V_Jac, V_Hessian):
    list_p = [z, Vbar]
    nz = z.shape[0]
    for i in range(nz):
        list_p.append(V_Jac[i])
    for i in range(nz):
        for j in range(nz):
            list_p.append(V_Hessian[(i,j)])
    return list_p

def verif_nlp(nlp, lbg, ubg, x, p, tol=1e-5, g_labels=None):
    """
        Function to verify if the nlp is feasible at a given solution point.
    """
    len_TR = tol
    ubg_val = ubg(len_TR).full().squeeze()
    lbg_val = lbg(len_TR).full().squeeze()

    g_fn = ca.Function("constraints", [nlp["x"], nlp["p"]], [nlp["g"]])
    g_val = g_fn(x, p).full().squeeze()

    upper_violation = np.max(g_val - ubg_val)
    lower_violation = np.max(lbg_val - g_val)

    if g_labels is not None:
        i_upper_violation = np.argmax(g_val - ubg_val)
        i_lower_violation = np.argmax(lbg_val - g_val)
        name_uviolation = g_labels[i_upper_violation]
        name_lviolation = g_labels[i_lower_violation]
    else:
        name_uviolation = " "
        name_lviolation = " "

    # print("upper_violation", upper_violation, "name", name_uviolation)
    assert upper_violation < tol, \
        f" Constraints violation in the upper bound of {name_uviolation}: {upper_violation:.2e}"
    assert lower_violation < tol, \
        f" Constraints violation in the lower bound of {name_lviolation}: {lower_violation:.2e}"