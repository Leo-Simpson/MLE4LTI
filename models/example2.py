import casadi as ca
def Model_example2():
    """
        Sensor fusion model
    """
    b = 0.02
    nalpha = 4
    nbeta = 4
    nx = 1
    ny = 2
    nu = 1
    ntheta = nalpha + nbeta

    cscale = 10
    dscale = 100
    rscale = 1.

    def dynamic(x, u, theta):
        alpha = theta[:nalpha]
        c1 = alpha[0] * cscale
        c2 = alpha[1] * cscale
        d1 = alpha[2] * dscale
        d2 = alpha[3] * dscale

        beta = theta[nalpha:]
        xplus = x + b * u
        y1 = c1 * x + d1
        y2 = c2 * x + d2

        y = ca.vcat([y1, y2])

        # noise model
        chol_Q = beta[0]
        chol_R = make_lower_triangular(ny, beta[1:])
        Q = chol_Q @ chol_Q.T
        R = chol_R @ chol_R.T * rscale


        # inequality constraints on the form h > 0
        h = ca.vertcat(ca.diag(chol_Q), ca.diag(chol_R))
        return xplus, y, Q, R, h

    # Define the model with Casadi symbolics
    x_symbol = ca.SX.sym("x", nx)
    u_symbol = ca.SX.sym("u", nu)
    theta_symbol = ca.SX.sym("theta", ntheta)
    xplus_symbol, y_symbol, Q_symbol, R_symbol, h_symbol = \
        dynamic(x_symbol, u_symbol, theta_symbol)
    # Define a Casadi functions associated with the model
    xplus_fn = ca.Function("xplus", [x_symbol, u_symbol, theta_symbol], [xplus_symbol])
    y_fn = ca.Function("y", [x_symbol, u_symbol, theta_symbol], [y_symbol])
    Q_fn = ca.Function("Q", [theta_symbol], [Q_symbol])
    R_fn = ca.Function("R", [theta_symbol], [R_symbol])
    h_fn = ca.Function("h", [theta_symbol], [h_symbol])
    model_dict = {
        "xplus_fn": xplus_fn,
        "y_fn": y_fn,
        "Q_fn": Q_fn,
        "R_fn": R_fn,
        "h_fn": h_fn
    }
    return model_dict

def make_lower_triangular(n, coeffs):
    """
    Construct a lower triangular matrix from a vector
    """
    ncoeffs = coeffs.shape[0]
    L = ca.SX.zeros(n, n)
    idx = 0
    for i in range(n):
        for j in range(i+1):
            assert idx < ncoeffs, "Not enough coefficients"
            L[i, j] = coeffs[idx]
            idx += 1
    return L