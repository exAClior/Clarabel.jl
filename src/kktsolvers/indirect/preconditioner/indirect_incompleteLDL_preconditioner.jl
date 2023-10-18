import LinearAlgebra: mul!, ldiv!
import LimitedLDLFactorizations: lldl
using SparseArrays

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
    y .= M.L \ x
    @. y = y / M.D
    y .= transpose(M.L) \ y
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

###################################################
# BlockIncompleteLDL preconditioner
###################################################
struct BlockIncompleteLDLPreconditioner{T,Tv} <: AbstractPreconditioner{T,Tv}

    L_full::SparseMatrixCSC{T,Int} # of form [[L1, 0], [0, L2]], where L1 is the L in ILDL factorization of A, L2 is the L in ILDL factorization of C
    D_full::AbstractVector{T}
    work::AbstractVector{T}

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function BlockIncompleteLDLPreconditioner{T,Tv}(
        m::Int,
        n::Int,
        KKT::SparseMatrixCSC{T,Int}
    ) where {T,Tv}
        
        dim = m+n
        L_full = sparse(I,dim,dim)
        D_full = Tv(zeros(T,dim))
        work = zeros(T,dim)

        return new(L_full, D_full, work)
    end
end

function ldiv!(
    y::AbstractVector{T},
    M::BlockIncompleteLDLPreconditioner{T},
    x::AbstractVector{T}
) where {T}
    # Use a series of solves on the IncompleteLDL factors.
    y .= M.L_full \ x
    @. y = y / M.D_full
    y .= transpose(M.L_full) \ y
end
ldiv!(M::BlockIncompleteLDLPreconditioner,x::AbstractVector) = ldiv!(x,M,x)

# TODO: ensure that this is correct.
function Base.size(M::BlockIncompleteLDLPreconditioner, idx)
    if(idx == 1 || idx == 2)
        return length(M.diagval)
    else
        error("Dimension mismatch error")
    end
end
Base.size(M::BlockIncompleteLDLPreconditioner) = (size(M,1),size(M,2))


# IncompleteLDL Preconditioning
function update_preconditioner!(
    solver::IndirectKKTSolver{T},
    preconditioner::BlockIncompleteLDLPreconditioner{T},
    L::SparseMatrixCSC{T,Int},
    D::AbstractVector{T}
) where {T}
    M = solver.M 
    A = solver.A
    work = M.work
    dim = length(work)
    m = solver.m
    p = solver.p

    # TODO
    # Take the top left and bottom right blocks
    @views top_left = solver.KKT[1:m, 1:m]
    @views bottom_right = solver.KKT[m+1:end, m+1:end]

    LLDL_top_left = lldl(top_left)
    LLDL_bottom_right = lldl(bottom_right)

    copyto!(M.L, blockdiag(LLDL_top_left.L, LLDL_bottom_right.L))
    copyto!(M.D, hcat(LLDL_top_left.D, LLDL_bottom_right.D))
end