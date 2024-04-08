import casadi as ca
import numpy as np
from scipy.linalg import sqrtm

class ModelParser:
    def __init__(self, Fdiscr, G, Q_fn, R_fn):
        self.Fdiscr = Fdiscr
        self.G =  G
        self.Q_fn = Q_fn
        self.R_fn = R_fn
        self.Ineq = lambda x : ca.DM(1.)  # default inequality constraint
        self.get_dim()

    def get_dim(self):
        self.ntheta = self.Fdiscr.size1_in(2)
        self.nx = self.Fdiscr.size1_in(0)
        self.nu = self.Fdiscr.size1_in(1)
        self.ny = self.G.size1_out(0)

    def feasible(self, theta):
        return self.Ineq(theta).full().min() > 0.

    def simulation(self, x0, us, theta, rng=None):
        # Q is the covariance of the noise which acts on the discrete equations
        if rng is None:
            rng = np.random.default_rng(seed=42)
        Q, R = self.Q_fn(theta), self.R_fn(theta)
        sqrt_R = sqrtm(R)
        sqrQ = sqrtm(Q)
        x = ca.DM(x0.copy())
        Ys = []
        Ytrues = []
        N = len(us)
        for i in range(N):
            ytrue = self.G(x, us[i], theta)
            y = ytrue + sqrt_R @ rng.normal(size=self.ny)
            Ys.append(y)
            Ytrues.append(ytrue)
            x = self.Fdiscr(x, us[i], theta)
            noise = sqrQ @  rng.normal(size=self.nx)
            x = x + noise

        ys = np.array([ y.full().reshape(-1) for y in Ys ])
        ys_true = np.array([ y.full().reshape(-1) for y in Ytrues ])
        return ys, ys_true

    def trajectory(self, x0, us, theta):
        x = x0.copy()
        xs = [x]
        ys = []
        for i in range(len(us)):
            y = self.G(x, us[i], theta)
            x = self.Fdiscr(x, us[i], theta)
            xs.append(x.full().squeeze())
            ys.append(y.full().squeeze())
        xs.pop() # the last x is not used
        return xs, ys

    def predictions(self, us, xs, theta, npred, Npred=None):
        N = len(us)
        ts = np.arange(N+1, dtype=int)
        ys_pred, ts_pred = [], []
        if Npred is None:
            Npred = int(float(N) / npred) # number of time point in one prediction interval
        interval = int(  float(N - Npred) / (npred - 1) )
        for i in range(npred):
            idx = i * interval
            xstart = xs[idx]
            upred = us[idx:idx+Npred]
            tpred = ts[idx:idx+Npred]
            _, ypred = self.trajectory(xstart, upred, theta)
            ys_pred.append(ypred)
            ts_pred.append(tpred)

        return ts_pred, ys_pred

    def get_linearization(self):
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        theta = ca.SX.sym("theta", self.ntheta)
        F = self.Fdiscr(x, u, theta)
        G = self.G(x, u, theta)

        x0 = np.zeros(self.nx)
        u0 = np.zeros(self.nu)
        A = ca.Function("Atemp", [x, u, theta], [ca.jacobian(F, x)])(x0, u0, theta)
        B = ca.Function("Btemp", [x, u, theta], [ca.jacobian(F, u)])(x0, u0, theta)
        C = ca.Function("Ctemp", [x, u, theta], [ca.jacobian(G, x)])(x0, u0, theta)
        D = ca.Function("Dtemp", [x, u, theta], [ca.jacobian(G, u)])(x0, u0, theta)
        b = ca.Function("btemp", [x, u, theta], [F])(x0, u0, theta)
        d = ca.Function("dtemp", [x, u, theta], [G])(x0, u0, theta)
        Q = self.Q_fn(theta)
        R = self.R_fn(theta)

        linearization =  ca.Function("linearization", [theta], [A, B, C, D, b, d, Q, R],
                                     ["theta"], ["A", "B", "C", "D", "b", "d", "Q", "R"]
                                     )

        return linearization
