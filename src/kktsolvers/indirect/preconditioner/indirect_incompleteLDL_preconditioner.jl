import LinearAlgebra: mul!, ldiv!
import LimitedLDLFactorizations
using SparseArrays

###################################################
# IncompleteLDL preconditioner
###################################################
struct IncompleteLDLPreconditioner{T,Tv} <: AbstractPreconditioner{T,Tv}

    LLDL::LimitedLDLFactorization{T}
    diagval::AbstractVector{T}

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function IncompleteLDLPreconditioner{T,Tv}(
        m::Int,
        n::Int,
        KKT::SparseMatrixCSC{T,Int}
    ) where {T,Tv}
        dim = m+n
        diagval = Tv(zeros(T,dim))
        # Generate sparse identity matrix
        initial_precond = sparse(1.0I, size(KKT, 1), size(KKT, 2))
        LLDL = lldl(initial_precond)

        return new(LLDL, diagval)
    end
end

function ldiv!(
    y::AbstractVector{T},
    M::IncompleteLDLPreconditioner{T},
    x::AbstractVector{T}
) where {T}
    y .= M.LLDL \ x
end
ldiv!(M::IncompleteLDLPreconditioner,x::AbstractVector) = ldiv!(x,M,x)

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
    diagval::AbstractVector{T}
) where {T}
    M = solver.M 

    new_LLDL = lldl(solver.KKT)
    # Take absolute value of diagonal values
    @. new_LLDL.D = abs(new_LLDL.D)

    # Update the preconditioner fields
    M.LLDL.n = new_LLDL.n # Int
    M.LLDL.colptr = new_LLDL.colptr # Vector
    M.LLDL.rowind = new_LLDL.rowind # Vector
    M.LLDL.Lrowind = new_LLDL.Lrowind # SubArray
    M.LLDL.lvals = new_LLDL.lvals # Vector
    M.LLDL.Lnzvals = new_LLDL.Lnzvals # SubArray
    M.LLDL.nnz_diag = new_LLDL.nnz_diag # Int
    M.LLDL.adiag = new_LLDL.adiag # Vector
    M.LLDL.D = new_LLDL.D # Vector
    M.LLDL.P = new_LLDL.P # V1
    M.LLDL.α = new_LLDL.α # T
    M.LLDL.α_increase_factor = new_LLDL.α_increase_factor # T
    M.LLDL.α_out = new_LLDL.α_out # T
    M.LLDL.memory = new_LLDL.memory # Int
    M.LLDL.Pinv = new_LLDL.Pinv # V2
    M.LLDL.wa1 = new_LLDL.wa1 # Vector
    M.LLDL.s = new_LLDL.s # Vector
    M.LLDL.w = new_LLDL.w # Vector
    M.LLDL.indr = new_LLDL.indr # Vector
    M.LLDL.indf = new_LLDL.indf # Vector
    M.LLDL.list = new_LLDL.list # Vector
    M.LLDL.pos = new_LLDL.pos # Vector
    M.LLDL.neg = new_LLDL.neg # Vector
    M.LLDL.computed_posneg = new_LLDL.computed_posneg # Bool
end


###################################################
# BlockIncompleteLDL preconditioner
###################################################
struct BlockIncompleteLDLPreconditioner{T,Tv} <: AbstractPreconditioner{T,Tv}

    LLDL_top::LimitedLDLFactorization{T} # for the top left block
    LLDL_bottom::LimitedLDLFactorization{T} # for the bottom right block
    diagval::AbstractVector{T}
    m::Int
    n::Int

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function BlockIncompleteLDLPreconditioner{T,Tv}(
        m::Int,
        n::Int,
        KKT::SparseMatrixCSC{T,Int}
    ) where {T,Tv}
        
        dim = m+n
        initial_precond_top = sparse(1.0I, m, m)
        initial_precond_bottom = sparse(1.0I, n, n)
        LLDL_top = lldl(initial_precond_top)
        LLDL_bottom = lldl(initial_precond_bottom)
        diagval = Tv(zeros(T,dim))

        return new(LLDL_top, LLDL_bottom, diagval, m, n)
    end
end

function ldiv!(
    y::AbstractVector{T},
    M::BlockIncompleteLDLPreconditioner{T},
    x::AbstractVector{T}
) where {T}
    # Solve the top left block
    y[1:M.m] .= M.LLDL_top \ x[1:M.m]
    # Solve the bottom right block
    y[M.m+1:end] .= M.LLDL_bottom \ x[M.m+1:end]
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
    diagval::AbstractVector{T}
) where {T}
    M = solver.M 
    A = solver.A
    m = solver.m
    p = solver.p

    # TODO
    # Take the top left and bottom right blocks
    top_left = solver.KKT[1:m, 1:m]
    bottom_right = solver.KKT[m+1:end, m+1:end]

    LLDL_top_new  = lldl(top_left)
    LLDL_bottom_new = lldl(bottom_right)

    # Take absolute value of diagonal values
    @. LLDL_top_new.D = abs(LLDL_top_new.D)
    @. LLDL_bottom_new.D = abs(LLDL_bottom_new.D)

    # Copy fields into the preconditioners
    M.LLDL_top.n = LLDL_top_new.n # Int
    M.LLDL_top.colptr = LLDL_top_new.colptr # Vector
    M.LLDL_top.rowind = LLDL_top_new.rowind # Vector
    M.LLDL_top.Lrowind = LLDL_top_new.Lrowind # SubArray
    M.LLDL_top.lvals = LLDL_top_new.lvals # Vector
    M.LLDL_top.Lnzvals = LLDL_top_new.Lnzvals # SubArray
    M.LLDL_top.nnz_diag = LLDL_top_new.nnz_diag # Int
    M.LLDL_top.adiag = LLDL_top_new.adiag # Vector
    M.LLDL_top.D = LLDL_top_new.D # Vector
    M.LLDL_top.P = LLDL_top_new.P # V1
    M.LLDL_top.α = LLDL_top_new.α # T
    M.LLDL_top.α_increase_factor = LLDL_top_new.α_increase_factor # T
    M.LLDL_top.α_out = LLDL_top_new.α_out # T
    M.LLDL_top.memory = LLDL_top_new.memory # Int
    M.LLDL_top.Pinv = LLDL_top_new.Pinv # V2
    M.LLDL_top.wa1 = LLDL_top_new.wa1 # Vector
    M.LLDL_top.s = LLDL_top_new.s # Vector
    M.LLDL_top.w = LLDL_top_new.w # Vector
    M.LLDL_top.indr = LLDL_top_new.indr # Vector
    M.LLDL_top.indf = LLDL_top_new.indf # Vector
    M.LLDL_top.list = LLDL_top_new.list # Vector
    M.LLDL_top.pos = LLDL_top_new.pos # Vector
    M.LLDL_top.neg = LLDL_top_new.neg # Vector
    M.LLDL_top.computed_posneg = LLDL_top_new.computed_posneg # Bool

    M.LLDL_bottom.n = LLDL_bottom_new.n # Int
    M.LLDL_bottom.colptr = LLDL_bottom_new.colptr # Vector
    M.LLDL_bottom.rowind = LLDL_bottom_new.rowind # Vector
    M.LLDL_bottom.Lrowind = LLDL_bottom_new.Lrowind # SubArray
    M.LLDL_bottom.lvals = LLDL_bottom_new.lvals # Vector
    M.LLDL_bottom.Lnzvals = LLDL_bottom_new.Lnzvals # SubArray
    M.LLDL_bottom.nnz_diag = LLDL_bottom_new.nnz_diag # Int
    M.LLDL_bottom.adiag = LLDL_bottom_new.adiag # Vector
    M.LLDL_bottom.D = LLDL_bottom_new.D # Vector
    M.LLDL_bottom.P = LLDL_bottom_new.P # V1
    M.LLDL_bottom.α = LLDL_bottom_new.α # T
    M.LLDL_bottom.α_increase_factor = LLDL_bottom_new.α_increase_factor # T
    M.LLDL_bottom.α_out = LLDL_bottom_new.α_out # T
    M.LLDL_bottom.memory = LLDL_bottom_new.memory # Int
    M.LLDL_bottom.Pinv = LLDL_bottom_new.Pinv # V2
    M.LLDL_bottom.wa1 = LLDL_bottom_new.wa1 # Vector
    M.LLDL_bottom.s = LLDL_bottom_new.s # Vector
    M.LLDL_bottom.w = LLDL_bottom_new.w # Vector
    M.LLDL_bottom.indr = LLDL_bottom_new.indr # Vector
    M.LLDL_bottom.indf = LLDL_bottom_new.indf # Vector
    M.LLDL_bottom.list = LLDL_bottom_new.list # Vector
    M.LLDL_bottom.pos = LLDL_bottom_new.pos # Vector
    M.LLDL_bottom.neg = LLDL_bottom_new.neg # Vector
    M.LLDL_bottom.computed_posneg = LLDL_bottom_new.computed_posneg # Bool
end


# The internal values for the choice of preconditioners
const PreconditionersDict = Dict([
    0           => Clarabel.NoPreconditioner,
    1           => Clarabel.DiagonalPreconditioner,
    2           => Clarabel.NormPreconditioner,
    3           => Clarabel.BlockDiagonalPreconditioner,
    4           => Clarabel.IncompleteLDLPreconditioner,
    5           => Clarabel.BlockIncompleteLDLPreconditioner
])