# Discussion

## 2024/01/26

#### Yuwen
**A)** Try [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) 
**B)** Try the Factorized Sparse Approximate Inverse (**FSAI**) (parallel) preconditioner 
**C)** Rough idea: Update the LDL factorization just in even iteration, use the initial factorized values as a warm-starting point to optimize (regarded as an inner optimization problem, similar idea to the parallel preconditioner). 