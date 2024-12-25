
using Krylov
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using LinearOperators

include("./indirect_defaults.jl")
include("../gpu/indirect_kkt_assembly.jl")
include("../gpu/indirect_datamaps.jl")
include("./indirect_minres.jl")
include("indirect_trimr.jl")