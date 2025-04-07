
<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-dark-jl.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-light-jl.png">
  <img alt="Clarabel.jl logo" src="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-light-jl.png" width="66%">
</picture>
<h1 align="center" margin=0px>
GPU implementation of Clarabel solver for Julia
</h1>
   <a href="https://github.com/oxfordcontrol/Clarabel.jl/actions"><img src="https://github.com/oxfordcontrol/Clarabel.jl/workflows/ci/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/oxfordcontrol/Clarabel.jl"><img src="https://codecov.io/gh/oxfordcontrol/Clarabel.jl/branch/main/graph/badge.svg"></a>
  <a href="https://oxfordcontrol.github.io/ClarabelDocs/stable"><img src="https://img.shields.io/badge/Documentation-stable-purple.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/oxfordcontrol/Clarabel.jl/releases"><img src="https://img.shields.io/badge/Release-v0.10.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license-">License</a> •
  <a href="https://oxfordcontrol.github.io/ClarabelDocs/stable">Documentation</a>
</p>

__CuClarabel.jl__ is the GPU implementation of the Clarabel solver, which can solve conic problems of the following form:

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
- __CuClarabel.jl__ can be added via the Julia package manager (type `]`): `pkg> dev https://github.com/cvxgrp/CuClarabel.git`, (which will overwrite current use of Clarabel solver).

## Tutorial

### Use in Julia
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

### Use in Python

We can call julia code within a python file by using [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) package. We can download the package by
```
pip install juliacall
```
Then, we load the package as a single variable `jl` which represents the Main module in Julia, and we can write Julia code and call it via `jl.seval()` in Python. 
```
from juliacall import Main as jl
import numpy as np
# Load Clarabel in Julia
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
jl.seval('using CUDA, CUDA.CUSPARSE')
```
Here we build up a simple optimization problem with a second-order cone, which is fully written by Julia.
```
jl.seval('''
    P = spzeros(3,3)
    q = [0, -1., -1]
    A = SparseMatrixCSC([1. 0 0; -1 0 0; 0 -1 0; 0 0 -1])
    b = [1, 0., 0., 0.]

    # 0-cone dimension 1, one second-order-cone of dimension 3
    cones = [Clarabel.ZeroConeT(1), Clarabel.SecondOrderConeT(3)]

    settings = Clarabel.Settings(direct_solve_method = :cudss)
                                    
    solver   = Clarabel.Solver(P, q, A, b, cones, settings)
    Clarabel.solve!(solver)
    
    # Extract solution
    x = solver.solution
''')
```
It is also possible to call the julia functions directly via JuliaCall. For example, if we want to reuse the solver object and update only coefficients in the problem, we can call the following blocks,
```
b_new = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float64)
jl.seval('b_gpu = CuVector{Float64,CUDA.UnifiedMemory}(b)')     #create a vector b_gpu that utilizes unified memory
jl.copyto_b(jl.b_gpu, b_new)                                    #directly copy a cpu vector b_new to a gpu vector b_gpu with unified memory

#############################################
# jl.seval('''
#     Clarabel.update_b!(solver,b)
#     Clarabel.solve!(solver)
# ''')
#############################################

jl.Clarabel.update_b_b(jl.solver,jl.b_gpu)          #Clarabel.update_b!()
jl.Clarabel.solve_b(jl.solver)                      #Clarabel.solve!()
```
where we create a vector `b_gpu` with unified memory that allows us copying value from a cpu-based vector `b_new` to gpu-based vectors. Note that we need to replace `!` in a julia function with `_b`. Reversely, we can also extract value from a Julia object back to Python,
```
# Retrieve the solution from Julia to Python
solution = np.array(jl.solver.solution.x)
print("Solution:", solution)
```
More examples can be found in the `example` folder.

### Performance tips
Due to the `just-in-time (JIT)` compilation in Julia, the first call of `CuClarabel` will also be slow in python and it is recommended to solve a mini problem first to trigger the JIT-compilation and get full performance on the subsequent solve of the actual problem. 

## Citing
```
@misc{CuClarabel,
      title={CuClarabel: GPU Acceleration for a Conic Optimization Solver}, 
      author={Yuwen Chen and Danny Tse and Parth Nobel and Paul Goulart and Stephen Boyd},
      year={2024},
      eprint={2412.19027},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2412.19027}, 
}
```

## License 🔍
This project is licensed under the Apache License  2.0 - see the [LICENSE.md](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/LICENSE.md) file for details.

