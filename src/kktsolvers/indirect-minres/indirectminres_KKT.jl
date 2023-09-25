struct KKTOperator{T}
    P::AbstractMatrix{T}
    A::AbstractMatrix{T}
    At::AbstractMatrix{T}
    H
    function KKTOperator(P::AbstractMatrix{T},A::AbstractMatrix{T}) where {T}
        At = SparseMatrixCSC(A')
        return new(P,A,At,H)
    end
end

function mul!(
    y::AbstractVector{T},
    KKT::KKTOperator{T},
    x::AbstractMatrix{T}
) where {T}
    (m,n) = size(KKT.A)

    
end