
using Krylov
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using LinearOperators

include("./indirect_defaults.jl")
include("./indirect_kkt_assembly.jl")
include("./indirect_datamaps.jl")
include("./indirect_minres.jl")
include("indirect_trimr.jl")