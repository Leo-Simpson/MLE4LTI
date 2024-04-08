# Nonlinear Programming for Maximum Likelihood Estimation of parameters in Linear Time Invariant systems.

This repository is an adaptation from the repository
[_"SQP-for-MLE-LTV"_](https://github.com/Leo-Simpson/SQP-for-MLE-LTV) for the special case of time-invariant linear systems, with a more efficient optimization routine.
The associated article (for the LTV version) is
[_"An Efficient Method for the Joint Estimation of System Parameters and
Noise Covariances for Linear Time-Variant Systems"_](https://arxiv.org/pdf/2211.12302.pdf)

## Prerequisites

Python packages
```
	- numpy
	- scipy
	- matplotlib
	- casadi (with IPOPT)
	- time
	- contextlib
```

# Description of the problem to solve

## Parametric Linear Time-Invariant Systems

We assume that the data is generated through the following dynamical system (linear, and with Gaussian Noise)

```math
\begin{equation}
	\begin{aligned}
		x_{k+1} &= A(\theta) x_k + B(\theta) u_k + b(\theta)+ w_k, && k = 0, \dots, N-1, \\
		y_{k} &= C(\theta) x_k + D(\theta) u_k + d(\theta)+ v_k, &&k = 0, \dots, N, \\
		w_k &\sim \mathcal{N}\left( 0, Q(\theta) \right), &&k = 0, \dots, N-1, \\
		v_k &\sim \mathcal{N}\left( 0, R(\theta) \right), &&k = 0, \dots, N, 
	\end{aligned}
\end{equation}
```
where $\theta$ are parameters of the model

Note that the time-varying behavior comes from the inputs $u_k$, which can appear in the dynamics linearly or nonlinearly (Note: this is also different from the paper, where the concept of inputs was not present).

Also, the parameters are assumed to be in some set defined with inequality constraints:

```math
\begin{align}
	\{ \theta \in \mathbb{R}^{n_{\theta}}  \; \big| \; h(\theta) \geq 0 \},
\end{align}
```

*Note that the inequality is of the opposite sign of how it is in the paper.*

## The Estimation

We consider optimization problems for estimation of $\theta$.
These are maximizing the performance of a Kalman filter on the training data over the parameters $\theta$.

### Prediction Error / Maximum Likelihood Estimation

```math
\begin{aligned}
	&\underset{ \substack{
			\theta, e,
			\hat{x}}, P, L, S,
		}
	{\mathrm{minimize}} \; \frac{1}{N}\sum_{k=1}^{N} F(e_k, S_k) \\
	& \mathrm{subject}  \, \mathrm{to} \, 
	\\&\qquad
	L_k = A( \theta) P_{k} \, C(\theta)^{\top} S_k^{-1}, 
	\\&\qquad
	S_k = C(\theta) \, P_{k} \, C(\theta)^{\top} + R(\theta), 
	\\&\qquad
	e_k = y_k - \left( C(\theta) \hat{x}_k + D(\theta) u_k + d(\theta) \right),
	\\&\qquad
	\hat{x}_{k+1} = A( \theta)\hat{x}_{k} + B(\theta)u_k + L_k e_k, 
	\\&\qquad
	P_{k+1} = A( \theta) P_kA( \theta)^{\top} - L_k S_k L_k^{\top} + Q(\theta) ,
	\\&\qquad
	h(\theta) \geq 0.
\end{aligned}
```

Regarding the cost function $F(\cdot, \cdot)$, two options are considered
```math
\begin{align}
	\begin{split}
		F_{\mathrm{MLE}}(e, S) & \equiv e^{\top} S^{-1} e + \log \det S, \\
		F_{\mathrm{PredErr}}(e, S) & \equiv \left\lVert e \right\rVert^2.
	\end{split}
\end{align}
```

The first of them is referred to as "MLE" because it corresponds to the Maximum Likelihood problem. 
The second is called "PredErr" because it corresponds to the Prediction Error Methods.

### Simplification for LTI

In the case of an LTI system, the Kalman Filter equations quickly converge to their steady state, which is given by the Riccati Equation.
Replacing the Kalman Filter equation with their corresponding steady states equation considerably reduces the complexity of the equations:

```math
\begin{aligned}
	&\underset{ \substack{
			\theta, e,
			\hat{x}}, P, L, S,
		}
	{\mathrm{minimize}} \; \frac{1}{N}\sum_{k=1}^{N} F(e_k, S) \\
	& \mathrm{subject}  \, \mathrm{to} \, 
	\\&\qquad
	e_k = y_k -  \left( C(\theta) \hat{x}_k + D(\theta) u_k + d(\theta) \right),
	\\&\qquad
	\hat{x}_{k+1} = A( \theta)\hat{x}_{k} + B(\theta)u_k + L e_k, 
	\\&\qquad
	L = A( \theta) P \, C(\theta)^{\top} S^{-1}, 
	\\&\qquad
	S = C(\theta) \, P \, C(\theta)^{\top} + R(\theta), 
	\\&\qquad
	P = A( \theta) P A( \theta)^{\top} - L S L^{\top} + Q(\theta) ,
	\\&\qquad
h(\theta) \geq 0.
\end{aligned}
```

### Reformulation for a better optimization
Let us parameterize $S$ and $P$ with additional variables $\eta$
(for example with the Cholesky decomposition to make it positive).

We also define the following:
```math
\begin{aligned}
	\tilde{A}(\theta, L) &=  A(\theta) - L C(\theta),\\
	\tilde{u}_k &= \begin{bmatrix} u_k \\ 1 \\ y_k\end{bmatrix}, \\
	\tilde{B}(\theta, L) &= 
	\begin{bmatrix} B(\theta) & b(\theta) - L d(\theta) & L \end{bmatrix}, \\
	\tilde{D}(\theta, L) &= 
	\begin{bmatrix} D(\theta) & d(\theta) & -I \end{bmatrix}, \\
	g(\theta, \eta, L) & =
	\begin{bmatrix}
		A( \theta) P(\eta) \, C(\theta)^{\top} - LS(\eta)
		\\
		C(\theta) \, P(\eta) \, C(\theta)^{\top} + R(\theta) - S(\eta)
		\\
		A( \theta) P(\eta) A( \theta)^{\top} - L S(\eta) L^{\top} + Q(\theta) - P(\eta)
	\end{bmatrix}, \\
	G_{MLE}(V, S) &= \mathrm{Trace}\left(V S^{-1}\right) + \log \det S, \\
	G_{PredErr}(V, S) &= \mathrm{Trace}\left(V \right).
\end{aligned}
```

Finally, we simply stack all main optimization variables $p = (\theta, \eta, L)$, which leads to the following formulation:
```math
\begin{aligned}
	&\underset{ \substack{
			p, e,
			\hat{x}}
		}
	{\mathrm{minimize}} \; G\left(\;  \frac{1}{N}\sum_{k=1}^{N} e_k e_k^\top, \; S(p) \right) \\
	& \mathrm{subject}  \, \mathrm{to} \, 
	\\&\qquad
	e_k = C(p) \hat{x}_k + \tilde{D}(p) \tilde{u}_k,
	\\&\qquad
	\hat{x}_{k+1} = \tilde{A}(p) \hat{x}_{k} + \tilde{B}(p) \tilde{u}_k, 
	\\&\qquad
	g(p) = 0,
	\\&\qquad
h(p) \geq 0 
\end{aligned}
```

where the function $G(\cdot, \cdot)$, is one of the two following
```math
\begin{align}
	\begin{split}
		G_{\mathrm{MLE}}(V, S) & \equiv \mathrm{Tr}\left( S^{-1} V \right) + \log \det S, \\
		G_{\mathrm{PredErr}}(V, S) & \equiv \mathrm{Tr}\left( V \right) .
	\end{split}
\end{align}
```


# Description of the algorithms

## IPOPT-lifted
One option is simply to call the solver IPOPT to solve the optimization problem in this lifted form:
```math
\begin{aligned}
	&\underset{ \substack{
			p, e,
			\hat{x}}
		}
	{\mathrm{minimize}} \; G\left(\;  \frac{1}{N}\sum_{k=1}^{N} e_k e_k^\top, \; S(p) \right) \\
	& \mathrm{subject}  \, \mathrm{to} \, 
	\\&\qquad
	e_k = C(p) \hat{x}_k + \tilde{D}(p) \tilde{u}_k,
	\\&\qquad
	\hat{x}_{k+1} = \tilde{A}(p) \hat{x}_{k} + \tilde{B}(p) \tilde{u}_k, 
	\\&\qquad
	g(p) = 0,
	\\&\qquad
	h(p) \geq 0,
\end{aligned}
```

## IPOPT-dense
One can also condense partially the problem to remove the dependency in $N$:

```math
\begin{aligned}
	&\underset{ p }{\mathrm{minimize}} \;
	G\left(\;  V(p), \; S(p) \right) \\
	& \mathrm{subject}  \, \mathrm{to} \, 
	\\&\qquad
	g(p) = 0,
	\\&\qquad
	h(p) \geq 0,
\end{aligned}
```

where the function $V(\cdot)$
is defined as follows: 
```math
\begin{aligned}
		V(p) & \coloneqq \frac{1}{N} \sum_{k=1}^{N} e_k(p) e_k(p)^\top
	\end{aligned}
```
and $e_k(p)$ are computed via the following equations:
```math
\begin{aligned}
	e_k &= C(p) \hat{x}_k + \tilde{D}(p) \tilde{u}_k,
	\\
	\hat{x}_{k+1} &= \tilde{A}(p) \hat{x}_{k} + \tilde{B}(p) \tilde{u}_k, 
\end{aligned}
```


## Sequential Programming (SP)

We propose a tailored SP method.
Here, we solve a sequence of optimization problems similar to the one above,
but where the function $V(\cdot)$ is approximated by a quadratic approximation to remove the dependency in the horizon $N$ in each optimization problem.
Ultimately, for globalization, a trust region approach is adopted, where the size of the trust region $\Delta$ is reduced when necessary.

In this algorithm, we update the current solution point $p^{(i)}$ by solving the following
(smaller) NonLinear Program:
```math
\begin{aligned}
	p^{(i+1)} = &\underset{ p }{\mathrm{\arg \min}} \;
	G\left(\;  V^{\textup{quad}}(p; p^{(i)}), \; S(p) \right) \\
	& \mathrm{subject}  \, \mathrm{to} \, 
	\\&\qquad
	g(p) = 0,
	\\&\qquad
	h(p) \geq 0,
	\\&\qquad
	\left\lVert p - p^{(i)} \right\rVert_1 \leq \Delta^{(i)},
\end{aligned}
```
where $V^{\textup{quad}}(p; \bar{p})$ is a (Gauss-Newton) quadratic approximation of $V(\cdot)$ around the point $\bar{p}$:
```math
\begin{aligned}
	V^{\textup{quad}}(p; \bar{p}) &\coloneqq
	\frac{1}{N}\sum_{k=1}^{N} e_k^{\textup{lin}}(p; \bar{p}) e_k^{\textup{lin}}(p; \bar{p})^\top,
\end{aligned}
```
where $e_k^{\textup{lin}}(p; \bar{p})$ is the linearization of $e_k(p)$:
```math
\begin{aligned}
	e_k^{\textup{lin}}(p; \bar{p})
	&\coloneqq
	e_k(\bar{p}) + \frac{d  e_k(\bar{p})}{d p }  (p  - \bar{p}).
\end{aligned}
```

Note that $e_k(\bar{p})$ and $\frac{d  e_k(\bar{p})}{d p }$ are computed via propagation of the following dynamical equations and of their derivatives
```math
\begin{aligned}
	e_k &= C(p) \hat{x}_k + \tilde{D}(p) \tilde{u}_k,
	\\
	\hat{x}_{k+1} &= \tilde{A}(p) \hat{x}_{k} + \tilde{B}(p) \tilde{u}_k.
\end{aligned}
```

Regarding the size of the trust region,
whenever the objective value of the new candidate point $p^{(i+1)}$ is higher the previous point $p^{(i)}$,
we reject the step and shrink the trust region: $\Delta^{(i+1)} = \gamma \Delta^{(i)}$.

# Benchmark

To assess the computational speed of the three algorithms,
we run them on a simple example, with generated data, for different realization of the data generation and different data lengths $N$.

## Example 1
For a simple model with $3$ states, $1$ input, $2$ outputs, representing heat transfers:
```math
\begin{aligned}
	\begin{bmatrix} x_1 \\ x_2 \\ x_3  \end{bmatrix}^{+}
	&=
	\begin{bmatrix} x_1 \\ x_2 \\ x_3  \end{bmatrix}
	+
	\theta_1 \begin{bmatrix} u - x_1 \\ x_1 - x_2 \\ x_2 - x_3  \end{bmatrix}
	+
	\theta_2
	\begin{bmatrix} 0 \\ 0 \\ 0 - x_3 \end{bmatrix} + w
	\\
	\begin{bmatrix} y_1 \\ y_2 \end{bmatrix}  &= \begin{bmatrix} x_1  \\  x_3  \end{bmatrix} + v
\end{aligned}
```
and with
```math
\begin{aligned}
	\mathbb{E}\big[ w w^\top\big] = Q(\theta) &= \begin{bmatrix} \theta_3 & 0 & 0 \\ 0 & \theta_3 & 0 \\ 0 & 0 & \theta_3  \end{bmatrix} \\
	\mathbb{E}\big[ v v^\top\big] = R(\theta) &= \begin{bmatrix} \theta_4 & 0 \\ 0 & \theta_4 \end{bmatrix} \\
\end{aligned}
```
The model is defined in [_models/example1_](https://github.com/Leo-Simpson/MLE4LTI/blob/main/models/example1.py),
and illustrated with generated data in [_notebooks/illustrative_example1_](https://github.com/Leo-Simpson/MLE4LTI/blob/main/notebooks/illustrative_example1.py).

## Example 2
For a different example, we consider now the following sensor fusion:
```math
\begin{aligned}
	x^{+} &= x + u + w
	\\
	\begin{bmatrix} y_1 \\ y_2 \end{bmatrix}  &= \begin{bmatrix} \alpha_1 x + \alpha_3 \\  \alpha_2 x + \alpha_4 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}
\end{aligned}
```
and with
```math
\begin{aligned}
	\mathbb{E}\big[ w w^\top\big] = Q(\theta) &= \beta_1^2 \\
	\mathbb{E}\big[ v v^\top\big] = R(\theta) &= \begin{bmatrix} \beta_2 & 0  \\ \beta_3 & \beta_4  \end{bmatrix}\begin{bmatrix} \beta_2 & 0  \\ \beta_3 & \beta_4  \end{bmatrix}^\top
\end{aligned}
```
The parameters to estimate are
```math
\begin{aligned}
	\theta &= \begin{bmatrix} \alpha \\ \beta \end{bmatrix}.
\end{aligned}
```

The model is defined in [_models/example2_](https://github.com/Leo-Simpson/MLE4LTI/blob/main/models/example2.py),
and illustrated with generated data in [_notebooks/illustrative_example1_](https://github.com/Leo-Simpson/MLE4LTI/blob/main/notebooks/illustrative_example2.py).

# Code example (how to use the package)

```python
'''
	xplus_fn: casadi function
	y_fn: casadi function
	Q_fn: casadi function
	R_fn: casadi function
	y1, y2 are arrays of measurement data with size N1 x ny and N2 x ny
	u1, u2 are arrays of input data with size N1 x nu and N2 x nu
	x0 is an array of the initial state of size nx
	theta0 initial guess to warm start the optimization routine
'''

from RiccatiEst import ModelParser # Model parser to define the model
from RiccatiEst import solve, compute_cost # main function to solve the problem

problem = {
    "model": ModelParser(xplus_fn, y_fn, Q_fn, R_fn),
    "ys": [y1, y2],
    "us": [u1, u2],
    "x0": x0 # can also be [x0_1, x0_2]
}

formulation = "MLE" # can be "MLE", "PredErr"
algorithm = "SP" # can be "SP" or "IPOPT-dense" or "IPOPT-lifted"

# default options for SP method (not necessary to specify them)
# when IPOPT is used, enter the ipopt options instead.
opts = {
  "TR_init":1.,               # initial lentgth of trust region \Delta
  "TR_shrink":0.5,            # = \gamma such that the trust region decreases as \Delta = \gamma \Delta when needed
  "maxiter":100,              # for termination
  "rtol.cost_decrease":1e-5,  # for termination
  "hessian_perturbation":0.,  # add \delta (p-\bar{p})^2 in the quadratic approximation of V
  "verbose":True              # printing information during optimization
  }
theta_found, stats = solve(problem, theta0, formulation, algorithm,
                                        opts=opts, verbose=True)
```

More detailed examples can be found in
[_notebooks/illustrative_example1_](https://github.com/Leo-Simpson/MLE4LTI/blob/main/notebooks/illustrative_example1.py)
and
[_notebooks/illustrative_example2_](https://github.com/Leo-Simpson/MLE4LTI/blob/main/notebooks/illustrative_example2.py).

