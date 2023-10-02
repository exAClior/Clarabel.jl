
using Krylov
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using LinearOperators

include("./indirect_defaults.jl")
# include("./preconditioner/cholesky_local.jl")
include("./indirect_kkt_assembly.jl")
include("./indirect_datamaps.jl")
include("./indirect_minres.jl")