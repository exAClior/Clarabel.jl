using LinearAlgebra, SparseArrays, Random
# using Clarabel
include("../../src\\Clarabel.jl")


T = Float64
rng = Random.MersenneTwister(242713)
n = 3
P = randn(rng,n,n)*1
P = diagm([0.3,2.0,1.6])
P = SparseMatrixCSC{T}(convert(Matrix{T},(P'*P)))
A = SparseMatrixCSC{T}(I(n)*one(T))
A1 = [A;-A]*2
c = T[0.1;-2.;1.]
b1 = ones(T,6)
cones = Clarabel.SupportedCone[Clarabel.NonnegativeConeT(3), Clarabel.NonnegativeConeT(3)]


#add a SOC constraint
A2 = SparseMatrixCSC{T}(I(n)*one(T))
b2 = [0;0;0]
A = [A1; A2]
b = [b1; b2]
push!(cones,Clarabel.SecondOrderConeT(3))

settings = Clarabel.Settings(
        equilibrate_enable=false,
        direct_kkt_solver=false,
        static_regularization_enable=false,
        presolve_enable=false
        )
        
solver   = Clarabel.Solver(P,c,A,b,cones,settings)
Clarabel.solve!(solver)