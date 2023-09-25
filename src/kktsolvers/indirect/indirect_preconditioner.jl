import LinearAlgebra: mul!, ldiv!

struct DiagonalPreconditioner{T} <: AbstractPreconditioner{T}

    diagval::AbstractVector{T}
    work::AbstractVector{T}
    m::Int              # the dimension of constraints
                        # YC: I think the augmented sparse form is no longer needed for the indirect method

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function DiagonalPreconditioner{T}(diagval::AbstractVector{T},m::Int) where {T}
        
        dim = length(diagval)
        work = ones(T,dim)

        return new(diagval,work,m)
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

function update_preconditioner(
    solver::AbstractIndirectSolver{T},
    diagval::AbstractVector{T}
) where {T}
    KKT = solver.KKTcpu
    M = solver.M 
    preconditioner = solver.preconditioner
    A = solver.A
    work = M.work
    dim = length(work)

    if (preconditioner == 1)
        m = M.m
        n = dim - m
        workm = @view work[(dim-m+1):dim]
        diagvalm = @view diagval[(dim-m+1):dim]
        workn = @view work[1:n]
        diagvaln = @view diagval[1:n]
        
        @. workm = -one(T)/diagvalm         # preconditioner for the constraint part, now assuming only linear inequality constraints

        #compute diagonals of A'*H^{-1}*A
        @. workn = zero(T)

        # # YC: computation of A'*H^{-1}*A, needs improvement
        # @inbounds for i= 1:m
        #     diaginv = -one(T)/diagvalm[i]

        #     Ati = KKT[1:n,n+i]
        #     @. workn += Ati*Ati*diaginv
        # end

        @inbounds Threads.@threads for i= 1:n
            tmp = zero(T)
            #traverse the column j in A
            @inbounds for j= A.colptr[i]:(A.colptr[i+1]-1)  
                tmp += A.nzval[j]^2*workm[A.rowval[j]]
            end
            workn[i] = tmp
        end

        @. workn = one(T)/(workn + diagvaln)      # preconditioner for the variable part
    end

    # Diagonal preconditioner, but we can make it more efficient
    if (preconditioner == 2)
        @. work = one(T)
        for i = 1:dim
            work[i] = one(T)/max(work[i],norm(KKT[:,i],Inf))
        end
    end

    @assert all(work .> zero(T))       #preconditioner need to be p.s.d.

    copyto!(M.diagval,work)
end

function check(

)
    #compute diagonals of A'*H^{-1}*A
    @. workn = zero(T)

    @inbounds for i= 1:m
        diaginv = -one(T)/diagvalm[i]

        #YC: computation of A'*H^{-1}*A, needs improvement
        Ati = KKT[1:n,n+i]
        @. workn += Ati*Ati*diaginv
    end
end