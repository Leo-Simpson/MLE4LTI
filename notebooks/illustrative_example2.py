# %%
# Leo Simpson, Tool-Temp AG, 2024

# %%
from IPython import get_ipython # type: ignore
ipython = get_ipython()
ipython.run_line_magic('load_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')
ipython.run_line_magic('matplotlib', 'ipympl')
# %%
import sys
import os
from os.path import join, dirname
main_dir = dirname(os.getcwd())
sys.path.append(main_dir)

# %%
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from RiccatiEst import ModelParser # Model parser to define the model
from RiccatiEst import solve # main function to solve the problem
from RiccatiEst import kalman # Kalman filter
from RiccatiEst import generate_u # utility function to generate inputs
from RiccatiEst import plot_data, plot_est # plotting tools
from models import Model_example2
from time import time
rng = np.random.default_rng(seed=42)

# %% [markdown]
# # Define the model
model_dict = Model_example2()
nalpha = 4
nbeta = 4
scales = {
    "u" : 0.2,
    "y" : 100
}

# %%
# Define true model
model_true = ModelParser(model_dict["xplus_fn"], model_dict["y_fn"],
                            model_dict["Q_fn"], model_dict["R_fn"])
alpha_true = np.array([5, -1, 0.2, 0.6]) # choose a true parameter
beta_true = np.array([1e-2, 1e-1, 1e-1, 1e-1])
theta_true = np.concatenate([alpha_true, beta_true])
print(theta_true)

# define x0
x0 = np.zeros(model_true.nx)
# %%
# generate u
Ntest = 3000
Ntrain = 1000
us_train = generate_u(
    rng, model_true.nu, Ntrain,
    umin=-scales["u"], umax=scales["u"], step_len = 200)

x0 = np.zeros(model_true.nx)
ys_train, _ = model_true.simulation(x0, us_train, theta_true, rng)

us_test = generate_u(
    rng, model_true.nu, Ntest,
    umin=-scales["u"], umax=scales["u"], step_len = 200)
ys_test, _ = model_true.simulation(x0, us_test, theta_true, rng)

# %%
fig1 = plot_data(us_train/scales["u"], ys_train/scales["y"])

# %% [markdown]
# # Estimation

# %%
# Here we use the model that is assumed
model = ModelParser(model_dict["xplus_fn"], model_dict["y_fn"],
                            model_dict["Q_fn"], model_dict["R_fn"])
model.Ineq = model_dict["h_fn"]
assert model.feasible(theta_true), "Constraints should be satisfied for the true parameters"

# %%
problemTrain = {
    "model": model,
    "ys": ys_train,
    "us": us_train,
    "x0": x0
}

# %%
# define the initial point for optimization
beta0 = np.ones(nbeta)
alpha0 = np.ones(nalpha)
theta0 = np.concatenate([alpha0, beta0])

# %% [markdown]
# ## Optimize over the Kalman filter

# %%
formulation = "MLE" # can be "MLE", "PredErr"
algorithm = "SP" # can be "SP" or "IPOPT-dense" or "IPOPT-lifted"
opts = {} # parameters
# %%
t0 = time()
theta_found, stats = solve(problemTrain, theta0, formulation, algorithm,
                                        opts=opts, verbose=True)
rtime = time() - t0

# %% [markdown]
# # Results
# %%
print(f"running time : {rtime:.2e}  status : {stats['return_status']}")

# %%
alpha_found = theta_found[:nalpha]
beta_found = theta_found[nalpha:]
# %%
print( alpha_true, alpha_found)
print(beta_true, beta_found)

# %%
e_alpha = np.linalg.norm((alpha_true-alpha_found)[:2])
e_beta = np.linalg.norm((beta_true-beta_found)[:2])
print(f"error on alpha : {e_alpha:.2e}   ,  error on beta : {e_beta:.2e}")

# %%
print(stats["return_status"])

# %% [markdown]
# ### Validation on out-of-sample data

# %%
xs_est, ys_est = kalman(model, x0, us_test, ys_test, theta_found)

npred = 3
t_pred, y_pred =  model.predictions(us_test, xs_est, theta_found, npred)

# %%
scale_y_over_u = scales["y"]/scales["u"]
fig =plot_est(us_test * scale_y_over_u, ys_test, ys_est, pred=(t_pred, y_pred))

# %%
