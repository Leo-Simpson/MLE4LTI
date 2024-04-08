import casadi as ca
def Model_example1():
    """
        Heat exchange model
    """
    dt = 1.
    beta_min = 1e-6
    Text = 0.
    alpha_max = 0.5

    nalpha = 2
    nbeta = 2

    nx = 3
    ny = 2
    nu = 1
    ntheta = nalpha + nbeta

    def dynamic(x, u, theta):
        alpha = theta[:nalpha]
        beta = theta[nalpha:]
        T1 = u[0]
        T2 = x[0]
        T3 = x[1]
        T4 = x[2]

        a_middle = alpha[0]
        aext = alpha[1]

        # equations of the system
        T2_plus = T2 + dt * a_middle * (T1 - T2)
        T3_plus = T3 + dt * a_middle * (T2 - T3)
        T4_plus = T4 + dt *( a_middle * (T3 - T4) + aext * (Text - T4))

        # construction of the output
        x_plus = ca.vcat([T2_plus, T3_plus, T4_plus])
        y = ca.vcat([T2, T4])

        # noise model
        Q =  beta[0] *  ca.DM.eye(nx)
        R =  beta[1] *  ca.DM.eye(ny)

        # inequality constraints on the form h > 0
        h = ca.vertcat(alpha, alpha_max - alpha, beta - beta_min)
        return x_plus, y, Q, R, h

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
