import LinearAlgebra: mul!, ldiv!

struct DiagonalPreconditioner{T} <: AbstractPreconditioner{T}

    diagval::AbstractVector{T}
    work::AbstractVector{T}

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function DiagonalPreconditioner{T}(diagval::AbstractVector{T}) where {T}
        
        dim = length(diagval)
        work = ones(T,dim)

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
