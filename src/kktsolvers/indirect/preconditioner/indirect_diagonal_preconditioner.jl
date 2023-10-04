import LinearAlgebra: mul!, ldiv!

###################################################
# No preconditioner
###################################################
struct NoPreconditioner{T,Tv} <: AbstractPreconditioner{T,Tv}
    function NoPreconditioner{T,Tv}(
        m::Int,
        n::Int,
        KKT::SparseMatrixCSC{T,Int}
    ) where {T,Tv}
        
        return I
    end

end

###################################################
# Diagonal preconditioner
###################################################
struct DiagonalPreconditioner{T,Tv} <: AbstractPreconditioner{T,Tv}

    diagval::AbstractVector{T}
    work::AbstractVector{T}

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function DiagonalPreconditioner{T,Tv}(
        m::Int,
        n::Int,
        KKT::SparseMatrixCSC{T,Int}
    ) where {T,Tv}
        
        dim = m+n
        work = zeros(T,dim)
        diagval = Tv(zeros(T,dim))

        return new(diagval,work)
    end

end

function ldiv!(
    y::AbstractVector{T},
    M::DiagonalPreconditioner{T},
    x::AbstractVector{T}
) where {T}
    @. y = M.diagval * x
end
ldiv!(M::DiagonalPreconditioner,x::AbstractVector) = ldiv!(x,M,x)

function Base.size(M::DiagonalPreconditioner, idx)
    if(idx == 1 || idx == 2)
        return length(M.diagval)
    else
        error("Dimension mismatch error")
    end
end
Base.size(M::DiagonalPreconditioner) = (size(M,1),size(M,2))


# Diagonal Preconditioning
function update_preconditioner!(
    solver::IndirectKKTSolver{T},
    preconditioner::DiagonalPreconditioner{T},
    diagval::AbstractVector{T}
) where {T}
    M = solver.M 
    A = solver.A
    work = M.work
    dim = length(work)
    m = solver.m
    p = solver.p

    #Diagonal partial preconditioner with diagonal approximation
    n = dim - m - p
    workn = @view work[1:n]
    diagvaln = @view diagval[1:n]
    workm = @view work[(n+1):(n+m)]
    diagvalm = @view diagval[(n+1):(n+m)]
    
    @. workm = -one(T)/diagvalm         # preconditioner for the constraint part, now assuming only linear inequality constraints
    #inverse the absolute value for augmented diagonals
    if (p > zero(T))
        workp = @view work[n+m+1:end]
        diagvalp = @view diagval[n+m+1:end]
        @. workp =  one(T)/abs(diagvalp) 
    end

    #compute diagonals of A'*H^{-1}*A
    @. workn = zero(T)

    @inbounds Threads.@threads for i= 1:n
        tmp = zero(T)
        #traverse the column j in A
        @inbounds for j= A.colptr[i]:(A.colptr[i+1]-1)  
            tmp += A.nzval[j]^2*workm[A.rowval[j]]
        end
        workn[i] = tmp
    end

    @. workn = one(T)/(workn + diagvaln)      # preconditioner for the variable part

    @assert all(work .> zero(T))       #preconditioner need to be p.s.d.

    copyto!(M.diagval,work)
end

###################################################
# Norm preconditioner
###################################################
struct NormPreconditioner{T,Tv} <: AbstractPreconditioner{T,Tv}

    diagval::AbstractVector{T}
    work::AbstractVector{T}

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function NormPreconditioner{T,Tv}(
        m::Int,
        n::Int,
        KKT::SparseMatrixCSC{T,Int}
    ) where {T,Tv}
        
        dim = m+n
        work = zeros(T,dim)
        diagval = Tv(zeros(T,dim))

        return new(diagval,work)
    end

end

function ldiv!(
    y::AbstractVector{T},
    M::NormPreconditioner{T},
    x::AbstractVector{T}
) where {T}
    @. y = M.diagval * x
end
ldiv!(M::NormPreconditioner,x::AbstractVector) = ldiv!(x,M,x)

function Base.size(M::NormPreconditioner, idx)
    if(idx == 1 || idx == 2)
        return length(M.diagval)
    else
        error("Dimension mismatch error")
    end
end
Base.size(M::NormPreconditioner) = (size(M,1),size(M,2))

function update_preconditioner!(
    solver::IndirectKKTSolver{T},
    preconditioner::NormPreconditioner{T},
    diagval::AbstractVector{T}
) where {T}
    KKT = solver.KKT
    M = solver.M 
    preconditioner = solver.settings.preconditioner
    work = M.work
    dim = length(work)

    # Diagonal norm preconditioner, but we can make it more efficient
    @. work = one(T)
    @inbounds Threads.@threads for i = 1:dim
        KKTcol = view(KKT.nzval,KKT.colptr[i]:(KKT.colptr[i+1]-1))
        work[i] = one(T)/max(work[i],norm(KKTcol,Inf))
    end

    @assert all(work .> zero(T))       #preconditioner need to be p.s.d.

    copyto!(M.diagval,work)
end



###############################################################
# Block diagonal preconditioner
###############################################################
mutable struct BlockDiagonalPreconditioner{T,Tv} <: AbstractPreconditioner{T,Tv}
    work::AbstractVector{T}
    n::Int
    Hinvblocks::SparseMatrixCSC{T,Int}
    F::Union{QDLDL.QDLDLFactorisation{T,Int}, Nothing}

    function BlockDiagonalPreconditioner{T,Tv}(
        m::Int,
        n::Int,
        KKT::SparseMatrixCSC{T,Int}
    ) where {T,Tv}
        
        dim = m+n
        work = ones(T,dim)
        F = nothing
        Hinvblocks = blockdiag(spdiagm(0 => ones(n)),KKT[n+1:n+m,n+1:n+m])  #YC: additional copy for the inverse of Hessian

        return new(work,n,Hinvblocks,F)
    end

end

function ldiv!(
    y::AbstractVector{T},
    M::BlockDiagonalPreconditioner{T},
    x::AbstractVector{T}
) where {T}
    copyto!(y,x)
    QDLDL.solve!(M.F,y)
end
ldiv!(M::BlockDiagonalPreconditioner,x::AbstractVector) = QDLDL.solve!(M.F,x)

function Base.size(M::BlockDiagonalPreconditioner, idx)
    if(idx == 1 || idx == 2)
        return length(M.F.Dinv.diag)
    else
        error("Dimension mismatch error")
    end
end
Base.size(M::BlockDiagonalPreconditioner) = (size(M,1),size(M,2))


function update_preconditioner!(
    kktsolver::IndirectKKTSolver{T},
    preconditioner::BlockDiagonalPreconditioner{T},
    diagval::AbstractVector{T}
) where {T}
    Hsblocks = kktsolver.Hsblocks
    Hinvblocks = preconditioner.Hinvblocks
    A = kktsolver.A
    ϵ = kktsolver.diagonal_regularizer
    n = kktsolver.n       #Update from the shift n + 1
    
    @views Hinvblocks.nzval[1:n] .= one(T)   #Preallocate values equal to 1

    ind = n
    for Hsi in Hsblocks
        len = length(Hsi)
        @views Hinvblocks.nzval[ind+1:ind+len] .= -one(T).*Hsi       #YC: reverse the sign change back
        ind += len
    end 

    @inbounds for i = (n+1):Hinvblocks.n 
        Hinvblocks[i,i] += ϵ
    end

    if preconditioner.F === nothing
        preconditioner.F = QDLDL.qdldl(Hinvblocks, perm = nothing)
    else
        preconditioner.F.workspace.triuA.nzval .= triu(Hinvblocks).nzval        #YC: for testing, should be optimized later
        QDLDL.refactor!(preconditioner.F)

        @assert all(isfinite, preconditioner.F.Dinv.diag)
    end

    #Update the first n diagonals
    workn = @view preconditioner.work[1:n]
    diagvaln = @view diagval[1:n]
    y1 = zeros(T,Hinvblocks.n)      #YC: redundant memory, should be removed later

    #compute diagonals of A'*H^{-1}*A
    @. workn = zero(T)

    @inbounds for i= 1:n
        Ai = @view A[:,i]
        @views y1[n+1:end] .= Ai
        
        #H^{-1} operator
        QDLDL.solve!(preconditioner.F,y1)
        workn[i] = dot(Ai,@view y1[n+1:end])
    end

    @views Hinvblocks.nzval[1:n] .= workn .+ diagvaln

    #update the first n diagonals and refactor the system
    @views preconditioner.F.workspace.triuA.nzval[1:n] .= Hinvblocks.nzval[1:n]        #YC: for testing, should be optimized later
    QDLDL.refactor!(preconditioner.F)           #Later, this can be simplified to update only the first n parts

    @assert all(preconditioner.F.Dinv.diag .> zero(T))       #preconditioner need to be p.s.d.
end

# The internal values for the choice of preconditioners
const PreconditionersDict = Dict([
    0           => Clarabel.NoPreconditioner,
    1           => Clarabel.DiagonalPreconditioner,
    2           => Clarabel.NormPreconditioner,
    3           => Clarabel.BlockDiagonalPreconditioner,
])