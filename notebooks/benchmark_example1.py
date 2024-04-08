# %%
# Leo Simpson, Tool-Temp AG, 2024

# %%
from IPython import get_ipython
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
from RiccatiEst import ModelParser # Model parser to define the model
from RiccatiEst import solve # main function to solve the problem
from RiccatiEst import generate_u # utility function to generate inputs
from RiccatiEst import latexify # utility function to set up plots
from models import Model_example1
from time import time
rng = np.random.default_rng(seed=42)

small = False
# %% [markdown]
# # Choose model
# %%
model_dict = Model_example1()
nalpha = 2
nbeta = 2
alpha_max = 0.5

# %% [markdown]
# # Generate data
def generate_data(N_, model_, theta, x0_, rng_):
    us_ = generate_u(rng_, model_.nu, N_, umax=50, step_len = 100)
    ys_, _ = model_.simulation(x0_, us_, theta, rng_)
    return us_, ys_
# %%
# Define true model
model_true = ModelParser(model_dict["xplus_fn"], model_dict["y_fn"],
                            model_dict["Q_fn"], model_dict["R_fn"])

# %%
# Define true parameters
rng = np.random.default_rng(123)
alpha_true = np.array([0.2, 0.4]) # choose a "true parameter randomly"
beta_true = np.array([1e-1, 1e-2])
theta_true = np.concatenate([alpha_true, beta_true])

# %% [markdown]
# # Estimation

# %%
# Here we use the model that is assumed
model = ModelParser(model_dict["xplus_fn"], model_dict["y_fn"],
                            model_dict["Q_fn"], model_dict["R_fn"])
model.Ineq = model_dict["h_fn"]
assert model.feasible(theta_true), "Constraints should be satisfied for the true parameters"

# Formulation
formulation = "MLE" # can be "MLE", "PredErr"
opts = {} # parameters
verbose = False
# %%
if small:
    NNs = 5
    Nmax = 5 * 1000
    nsamples = 5
else:
    NNs = 10
    Nmax = 10 * 1000
    nsamples = 10

algorithms = ["SP", "IPOPT-dense", "IPOPT-lifted"]
Ns = np.geomspace(100, Nmax, num=NNs, dtype=int)
dicts = {
    algo : {N:[] for N in Ns}
    for algo in algorithms
}
for iN, N in enumerate(Ns):
    print(f"Number of data points : {N} ({iN+1}/{len(Ns)})")
    for i in range(nsamples):
        print(f"   Sample {i+1}/{nsamples}")
        x0 = np.zeros(model.nx)
        us, ys = generate_data(N, model_true, theta_true, x0, rng)
        problem = {
            "model": model,
            "ys": ys,
            "us": us,
            "x0": x0
        }
        # Initial point
        alpha0 = rng.random(nalpha)* alpha_max
        beta0 = rng.random(nalpha)
        theta0 = np.concatenate([alpha0, beta0])
        for algorithm in algorithms:
            if N > 5000 and algorithm == "IPOPT-lifted": # takes forever
                continue
            print(f"     Algorithm : {algorithm}")
            t0 = time()
            theta_found, stats = solve(problem, theta0, formulation, algorithm,
                                                    opts=opts, verbose=verbose)
            rtime = time() - t0
            info = {
                "theta_found" : theta_found,
                "stats" : stats,
                "rtime" : rtime
            }
            dicts[algorithm][N].append(info)
            print(f'''
                running time : {rtime:.2f}
                status : {stats['return_status']}
                cost : {stats['cost']:.5e}
            ''')
# %%
for algorithm in algorithms:
    for N in Ns:
        if len(dicts[algorithm][N]) == 0:
            del dicts[algorithm][N]
# %%
colors = {
    "SP" : "green",
    "IPOPT-dense" : "orange",
    "IPOPT-lifted" : "purple"
}

latexify()
fig, ax = plt.subplots(1, figsize=(8, 6))
ax.set_xlabel(r"Number of data points $N$")
ax.set_ylabel("Runtime (s)")
ax.grid()
ax.set_xscale("log")
ax.set_yscale("log")
for algorithm in algorithms:
    Ns = list(dicts[algorithm].keys())
    rtimes = [
            [dicts[algorithm][N][i]["rtime"] for i in range(nsamples)]
            for N in Ns
    ]
    rtimes_mean = np.mean(rtimes, axis=1)
    rtimes_max = np.max(rtimes, axis=1)
    rtimes_min = np.min(rtimes, axis=1)
    ax.plot(Ns, rtimes_mean, "-", label=algorithm, color=colors[algorithm])
    ax.plot(Ns, rtimes, ".", markersize=3, color=colors[algorithm])
    ax.fill_between(Ns, rtimes_min, rtimes_max, alpha=0.2, color=colors[algorithm])
ax.legend()
# %%
plot_dir = join(main_dir, "plots")
file_name = "benchmark_example1"
if small:
    file_name = "small_" + file_name
name = join(plot_dir, file_name)
fig.savefig(f"{name}.png", bbox_inches='tight', pad_inches=0.02)
# %%
