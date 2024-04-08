import casadi as ca
import numpy as np

irrelevant_options = [
            "TR_init",
            "TR_shrink",
            "maxiter",
            "rtol.cost_decrease",
            "verbose",
            "hessian_perturbation",
        ]

class NLP:
    def __init__(self):
        self.nlp = {
            "f": 0., 
            "x": [],
            "g": []
            }

        self.solver_dict = {
            "x0": [],
            "lbx": [],
            "ubx": [],
            "lbg": [],
            "ubg": [],
        }
        self.g_labels = []
        self.nlpsolver_options = {}
        self.nlpsol = None
        self.retrieve = lambda x: x
        self.opts = {}

    def add_var(self, var, var0, ubx=np.inf, lbx=-np.inf):
        self.nlp["x"].append(var.T.reshape((-1, 1)))
        if not isinstance(var0, np.ndarray):
            var0 = var0.full()
        var0_ = var0.reshape(-1)
        self.solver_dict["x0"].append(var0_)
        self.solver_dict["lbx"].append(np.ones(len(var0_))*lbx)
        self.solver_dict["ubx"].append(np.ones(len(var0_)) * ubx)

    def add_eq(self, eq, ubg=0., lbg=0., names=None):
        eq_ = eq.reshape((-1, 1))
        n = eq_.shape[0]
        self.nlp["g"].append(eq_)
        self.solver_dict["lbg"].append(np.ones(n) * lbg)
        self.solver_dict["ubg"].append(np.ones(n) * ubg)

        if names is None:
            names = ["unnamed" for i in range(n)]
        else:
            assert len(names) == n, "names should have the same length as the constraint"
        self.g_labels = self.g_labels + names

    def add_value(self, value):
        self.nlp["f"] = self.nlp["f"] + value

    def verif_init(self, tol=1e-8):
        g0 = ca.Function("constr", [self.nlp["x"]], [self.nlp["g"]])(self.solver_dict["x0"])
        g0_arr = g0.full().squeeze()

        upper_violation = np.max(g0_arr - self.solver_dict["ubg"])
        lower_violation = np.max(self.solver_dict["lbg"] - g0_arr)

        i_upper_violation = np.argmax(g0_arr - self.solver_dict["ubg"])
        i_lower_violation = np.argmax(self.solver_dict["lbg"] - g0_arr)

        name_uviolation = self.g_labels[i_upper_violation]
        name_lviolation = self.g_labels[i_lower_violation]

        assert upper_violation < tol, \
            f" Constraints violation in the upper bound of {name_uviolation}: {upper_violation:.2e}"
        assert lower_violation < tol, \
            f" Constraints violation in the lower bound of {name_lviolation}: {lower_violation:.2e}"

    def _solver_options(self):
        self.nlpsolver_options = {
            "expand": False,
            "ipopt.max_iter": 500,
            "ipopt.max_cpu_time": 120.0,
            "ipopt.linear_solver": "mumps",  # suggested: ma57
            "ipopt.mumps_mem_percent": 10000,
            "ipopt.mumps_pivtol": 0.001,
            "ipopt.print_level": 5,
            "ipopt.print_frequency_iter": 10
        }
        for key, value in self.opts.items():
            if key not in irrelevant_options:
                self.nlpsolver_options[key] = value

    def presolve(self):
        self._solver_options()
        self.nlpsol = ca.nlpsol("S", "ipopt", self.nlp, self.nlpsolver_options)

    def final_solve(self):
        r = self.nlpsol(
            x0=self.solver_dict["x0"],
            lbx=self.solver_dict["lbx"],
            ubx=self.solver_dict["ubx"],
            lbg=self.solver_dict["lbg"],
            ubg=self.solver_dict["ubg"],
        )
        return r

    def stack(self):
        self.nlp["x"] = ca.vcat(self.nlp["x"])
        self.nlp["g"] = ca.vcat(self.nlp["g"])
        for key, item in self.solver_dict.items():
            self.solver_dict[key] = np.concatenate(item)

    def define_important_variable(self, var):
        self.retrieve = ca.Function("retrieve", [ca.vcat(self.nlp["x"])], [var])

    def solve(self):
        self.presolve()
        full_solution = self.final_solve()
        important_solution = self.retrieve(full_solution["x"]).full().squeeze()
        stats = self.nlpsol.stats()
        stats["cost"] = full_solution["f"].full().squeeze()
        return important_solution, stats
