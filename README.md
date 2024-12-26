
<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/oxfordcontrol/ClarabelDocs/blob/main/docs/src/assets/logo-banner-dark-jl.png" width=60%>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/oxfordcontrol/ClarabelDocs/blob/main/docs/src/assets/logo-banner-light-jl.png" width=60%>
  <img alt="Clarabel.jl logo" src="https://github.com/oxfordcontrol/ClarabelDocs/blob/main/docs/src/assets/logo-banner-light-jl.png" height="25">
</picture>
<h1 align="center" margin=0px>
GPU implementation of Clarabel solver for Julia
</h1>
   <a href="https://github.com/oxfordcontrol/Clarabel.jl/actions"><img src="https://github.com/oxfordcontrol/Clarabel.jl/workflows/ci/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/oxfordcontrol/Clarabel.jl"><img src="https://codecov.io/gh/oxfordcontrol/Clarabel.jl/branch/main/graph/badge.svg"></a>
  <a href="https://oxfordcontrol.github.io/ClarabelDocs/stable"><img src="https://img.shields.io/badge/Documentation-stable-purple.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/oxfordcontrol/Clarabel.jl/releases"><img src="https://img.shields.io/badge/Release-v0.9.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license-">License</a> •
  <a href="https://oxfordcontrol.github.io/ClarabelDocs/stable">Documentation</a>
</p>

__clarabel-gpu.jl__ is the GPU implementation of the Clarabel solver, which can solve conic problems of the following form:

$$
\begin{array}{r}
\text{minimize} & \frac{1}{2}x^T P x + q^T x\\\\[2ex]
\text{subject to} & Ax + s = b \\\\[1ex]
        & s \in \mathcal{K}
\end{array}
$$

with decision variables
$x \in \mathbb{R}^n$,
$s \in \mathbb{R}^m$
and data matrices
$P=P^\top \succeq 0$,
$q \in \mathbb{R}^n$,
$A \in \mathbb{R}^{m \times n}$, and
$b \in \mathbb{R}^m$.
The set $\mathcal{K}$ is a composition of convex cones; we support zero cones (linear equality constraints), nonnegative cones (linear inequality constraints), second-order cones, exponential cone and power cones. Our package relies on the external package [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) for the linear system solver [CUDSS](https://developer.nvidia.com/cudss). We also support linear system solves in lower (mixed) precision.


## Installation
- __clarabel-gpu.jl__ can be added via the Julia package manager (type `]`): `pkg> dev https://github.com/cvxgrp/clarabel-gpu.git`, (which will overwrite current use of Clarabel solver).

## Tutorial
Modeling a conic optimization problem is the same as in the original [Clarabel solver](https://clarabel.org/stable/), except with the additional parameter `direct_solve_method`. This can be set to `:cudss` or `:cudssmixed`. Here is a portfolio optimization problem modelled via JuMP:
```
using LinearAlgebra, SparseArrays, Random, JuMP
using Clarabel

## generate the data
rng = Random.MersenneTwister(1)
k = 5; # number of factors
n = k * 10; # number of assets
D = spdiagm(0 => rand(rng, n) .* sqrt(k))
F = sprandn(rng, n, k, 0.5); # factor loading matrix
μ = (3 .+ 9. * rand(rng, n)) / 100. # expected returns between 3% - 12%
γ = 1.0; # risk aversion parameter
d = 1 # we are starting from all cash
x0 = zeros(n);

a = 1e-3
b = 1e-1
γ = 1.0;

model = JuMP.Model(Clarabel.Optimizer)
set_optimizer_attribute(model, "direct_solve_method", :cudss)

@variable(model, x[1:n])
@variable(model, y[1:k])   
@variable(model, s[1:n])
@variable(model, t[1:n])
@objective(model, Min, x' * D * x + y' * y - 1/γ * μ' * x);
@constraint(model, y .== F' * x);
@constraint(model, x .>= 0);

# transaction costs
@constraint(model, sum(x) + a * sum(s) == d + sum(x0) );
@constraint(model, [i = 1:n], x0[i] - x[i] == t[i]) 
@constraint(model, [i = 1:n], [s[i], t[i]] in MOI.SecondOrderCone(2));
JuMP.optimize!(model)
```

## Citing
```
Coming out soon
```

## License 🔍
This project is licensed under the Apache License  2.0 - see the [LICENSE.md](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/LICENSE.md) file for details.

