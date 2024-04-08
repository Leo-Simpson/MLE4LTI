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
from RiccatiEst import ModelParser # Model parser to define the model
from RiccatiEst import solve, compute_cost # main function to solve the problem
from RiccatiEst import kalman # Kalman filter
from RiccatiEst import generate_u # utility function to generate inputs
from RiccatiEst import plot_data, plot_est # plotting tools
from models import Model_example1
from time import time
rng = np.random.default_rng(seed=42)

# %% [markdown]
# # Define the model
model_dict = Model_example1()
nalpha = 2
nbeta = 2
alpha_max = 0.5

# %%
# Define true model
model_true = ModelParser(model_dict["xplus_fn"], model_dict["y_fn"],
                            model_dict["Q_fn"], model_dict["R_fn"])

alpha_true = np.array([0.2, 0.4]) # choose a true parameter
beta_true = np.array([1e-1, 1e-2])
theta_true = np.concatenate([alpha_true, beta_true])
print(theta_true)

# define x0
x0 = np.zeros(model_true.nx)
# %%
# generate u
Ntest = 100
Ntrain = 1000
umax = 50

us_train1 = generate_u(rng, model_true.nu, Ntrain, umax=umax, step = 10)
us_train2 = generate_u(rng, model_true.nu, Ntrain, umax=umax, step = 10)
us_test = generate_u(rng, model_true.nu, Ntest, umax=umax, step = 10)

# generate y by simulating the system
ys_train1, _ = model_true.simulation(x0, us_train1, theta_true, rng)
ys_train2, _ = model_true.simulation(x0, us_train2, theta_true, rng)
ys_test, _ = model_true.simulation(x0, us_test, theta_true, rng)

# %%
fig1 = plot_data(us_train1, ys_train1)
fig1 = plot_data(us_train2, ys_train2)

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
    "ys": [ys_train1, ys_train2],
    "us": [us_train1, us_train2],
    "x0": x0
}

# %%
# define the initial point for optimization
beta0 = np.ones(nbeta)
alpha0 = rng.random(nalpha)* alpha_max
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
cost = compute_cost(problemTrain, theta_found, formulation)
print(f'''
      running time : {rtime:.2e}
      status : {stats['return_status']}
      cost : {cost:.2e}
    ''')
# %%
alpha_found = theta_found[:2]
beta_found = theta_found[2:]
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
fig = plot_est(us_test, ys_test, ys_est, pred=(t_pred, y_pred))

# %%
