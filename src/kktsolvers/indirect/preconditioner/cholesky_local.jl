function factor!(
    cones::CompositeCone{T},
    Hinvblock::Vector{Vector{T}},
    Hsblock::Vector{Vector{T}},
    regularizer::T
) where {T}
    for (cone, invblock, block) in zip(cones,Hinvblock,Hsblock)
        @conedispatch factor!(cone,invblock,block,regularizer)
    end
    return nothing
end

function ldiv!(
    cones::CompositeCone{T},
    y::ConicVector{T},
    Hinvblock::Vector{Vector{T}},
    x::ConicVector{T}
) where {T}
    for (cone, yi, invblock, xi) in zip(cones,y.views,Hinvblock,x.views)
        @conedispatch ldiv!(cone,yi,invblock,xi)
    end
    return nothing
end

#############################################
function factor!(
    cone::Union{ZeroCone{T},NonnegativeCone{T}},
    Hinvblock::Vector{T},
    Hsblock::Vector{T},
    regularizer::T
) where {T}
    @. Hinvblock = inv(Hsblock + regularizer)
end

function factor!(
    cone::Union{SecondOrderCone{T},ExponentialCone{T},PowerCone{T}},
    Hinvblock::Vector{T},
    Hsblock::Vector{T},
    regularizer::T
) where {T}
    Hs = reshape(Hsblock, (3,3))
    @inbounds for i = 1:3
        Hs[i,i] += regularizer
    end
    Hinv = reshape(Hinvblock, (3,3))
    cholesky_3x3_explicit_factor!(Hinv,Hs)
end

function ldiv!(
    cone::Union{ZeroCone{T},NonnegativeCone{T}},
    y::AbstractVector{T},
    Hinvblock::Vector{T},
    x::AbstractVector{T}
) where {T}
    @. y = Hinvblock*x
end

function ldiv!(
    cone::Union{SecondOrderCone{T}, ExponentialCone{T}, PowerCone{T}},
    y::AbstractVector{T},
    Hinvblock::Vector{T},
    x::AbstractVector{T}
) where {T}
    Hinv = reshape(Hinvblock,(3,3))
    cholesky_3x3_explicit_solve_2!(y,Hinv,x)
end