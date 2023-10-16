import LinearAlgebra: mul!, ldiv!
import LimitedLDLFactorizations: lldl

###################################################
# IncompleteLDL preconditioner
###################################################
struct IncompleteLDLPreconditioner{T,Tv} <: AbstractPreconditioner{T,Tv}

    L::SparseMatrixCSC{T,Int}
    D::AbstractVector{T}
    work::AbstractVector{T}

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function IncompleteLDLPreconditioner{T,Tv}(
        m::Int,
        n::Int,
        KKT::SparseMatrixCSC{T,Int}
    ) where {T,Tv}
        
        dim = m+n
        L = sparse(I,dim,dim)
        D = Tv(zeros(T,dim))
        work = zeros(T,dim)

        return new(L, D, work)
    end
end

function ldiv!(
    y::AbstractVector{T},
    M::IncompleteLDLPreconditioner{T},
    x::AbstractVector{T}
) where {T}
    # Use a series of solves on the IncompleteLDL factors.
    y .= transpose(M.L) \ x
    y .= y \ M.D
    y .= M.L \ y
end
ldiv!(M::IncompleteLDLPreconditioner,x::AbstractVector) = ldiv!(x,M,x)

# TODO: ensure that this is correct.
function Base.size(M::IncompleteLDLPreconditioner, idx)
    if(idx == 1 || idx == 2)
        return length(M.diagval)
    else
        error("Dimension mismatch error")
    end
end
Base.size(M::IncompleteLDLPreconditioner) = (size(M,1),size(M,2))


# IncompleteLDL Preconditioning
function update_preconditioner!(
    solver::IndirectKKTSolver{T},
    preconditioner::IncompleteLDLPreconditioner{T},
    L::SparseMatrixCSC{T,Int},
    D::AbstractVector{T}
) where {T}
    M = solver.M 
    A = solver.A
    work = M.work
    dim = length(work)
    m = solver.m
    p = solver.p

    # Take the LLDL transform of the KKT matrix
    LLDL = lldl(solver.KKT)

    copyto!(M.L, LLDL.L)
    copyto!(M.D, LLDL.D)
end