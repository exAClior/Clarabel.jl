mutable struct TRIMRIndirectSolver{T} <: AbstractIndirectSolver{T}

    solver#::TrimrSolver{T,T,AbstractVector{T}}
    KKTcpu::SparseMatrixCSC{T,Int}
    At::AbstractSparseMatrix{T}
    Fp::Union{QDLDL.QDLDLFactorisation{T,Int}, Nothing}     #1x1 block
    Fh::Union{QDLDL.QDLDLFactorisation{T,Int}, Nothing}     #2x2 block

    b::AbstractVector{T}
    c::AbstractVector{T}

    #YC: additional parameter
    m::Int
    n::Int
    cpu::Bool
    atol::T                         #YC: using iterative refinement atol and rtol values at present
    rtol::T

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function TRIMRIndirectSolver{T}(KKT0::SparseMatrixCSC{T,Int}, A::SparseMatrixCSC{T,Int}, settings) where {T}
        
        cpu = (settings.device == :cpu) ? true : false;
        Vectype = (cpu) ? Vector : CuVector;
        (m,n) = size(A)

        #YC: KKTcpu shares the same memory with KKT0
        At = (cpu) ? SparseMatrixCSC(A') : CuSparseMatrixCSR(A');     #We use CUDA if gpu is selected

        KKTcpu = KKT0
        Fp = nothing
        Fh = nothing
        
        b = Vectype(ones(T,n))
        c = Vectype(ones(T,m))

        solver = TrimrSolver(n,m,Vectype{T})

        atol = settings.iterative_refinement_abstol
        rtol = settings.iterative_refinement_reltol

        return new(solver,KKTcpu,At,
                Fp,Fh,b,c,m,n,
                cpu,atol,rtol)
    end

end

IndirectSolversDict[:trimr] = TRIMRIndirectSolver
required_matrix_shape(::Type{TRIMRIndirectSolver}) = :full

#refactor the linear system
function refactor!(trimrsolver::TRIMRIndirectSolver{T}) where{T}

    KKTcpu = trimrsolver.KKTcpu
    n = trimrsolver.n

    trimrsolver.Fp = QDLDL.qdldl(KKTcpu[1:n,1:n], perm = nothing)
    trimrsolver.Fh = QDLDL.qdldl(-KKTcpu[n+1:end,n+1:end], perm = nothing)

    return is_success = true # required to compile for some reason

end

#####################################
import LinearAlgebra: ldiv!
function ldiv!(
    y::AbstractVector{T},
    M::QDLDL.QDLDLFactorisation{T,Int},
    x::AbstractVector{T}
) where {T}
    copyto!(y,x)
    QDLDL.solve!(M,y)
end
#####################################

#solve the linear system
function solve!(
    trimrsolver::TRIMRIndirectSolver{T},
    preconditioner::Union{AbstractPreconditioner{T},UniformScaling{Bool}},
    x::AbstractVector{T},
    b::AbstractVector{T}
) where{T}

    At = trimrsolver.At
    m = trimrsolver.m
    n = trimrsolver.n
    Fp = trimrsolver.Fp
    Fh = trimrsolver.Fh

    #data copy to gpu
    copyto!(trimrsolver.b,@view b[1:n])
    copyto!(trimrsolver.c,@view b[n+1:end])

    #Preconditioner
    trimr!(trimrsolver.solver,At, trimrsolver.b, trimrsolver.c; M=Fp, N=Fh, ldiv = true)#,verbose= Int64(floor(length(b)/10)),history=true)  # verbose=n,history=true

    # trimrsolver.iter += trimrsolver.solver.stats.niter
    # println("minres error: ", trimrsolver.solver.stats.niter)

    # println("Iter num", trimrsolver.solver.stats.niter)
    x[1:n] .= trimrsolver.solver.x
    x[n+1:end] .= trimrsolver.solver.y

    # todo: note that the second output of minres is the stats, perhaps might be useful to the user

end