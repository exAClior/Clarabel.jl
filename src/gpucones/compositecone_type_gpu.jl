using CUDA, CUDA.CUSPARSE

# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------

struct CompositeConeGPU{T} <: AbstractCone{T}
    #YC: redundant CPU data, need to be removed later
    cones::AbstractVector{AbstractCone{T}}  

    #count of each cone type
    type_counts::Dict{Type,Cint}

    #overall size of the composite cone
    numel::Cint
    degree::Cint

    #range views
    rng_cones::AbstractVector{UnitRange{Cint}}
    rng_blocks::AbstractVector{UnitRange{Cint}}

    # the flag for symmetric cone check
    _is_symmetric::Bool
    n_linear::Cint

    idx_eq::Vector{Cint}
    idx_inq::Vector{Cint}

    #data
    w::AbstractVector{T}
    λ::AbstractVector{T}
    η::AbstractVector{T}

    #nonsymmetric cone
    αp::AbstractVector{T}           #power parameters of power cones
    H_dual::AbstractArray{T}        #Hessian of the dual barrier at z 
    Hs::AbstractArray{T}            #scaling matrix
    grad::AbstractArray{T}         #gradient of the dual barrier at z 

    #step_size
    α::AbstractVector{T}

    function CompositeConeGPU{T}(cpucones::CompositeCone{T}) where {T}

        #Information from the CompositeCone on CPU 
        cones  = cpucones.cones
        type_counts = cpucones.type_counts
        _is_symmetric = cpucones._is_symmetric

        n_zero = haskey(type_counts,ZeroCone) ? type_counts[ZeroCone] : 0
        n_nn = haskey(type_counts,NonnegativeCone) ? type_counts[NonnegativeCone] : 0
        n_linear = n_zero + n_nn
        n_soc = haskey(type_counts,SecondOrderCone) ? type_counts[SecondOrderCone] : 0
        n_exp = haskey(type_counts,ExponentialCone) ? type_counts[ExponentialCone] : 0
        n_pow = haskey(type_counts,PowerCone) ? type_counts[PowerCone] : 0
        #idx set for eq and ineq constraints
        idx_eq = Vector{Cint}([])
        idx_inq = Vector{Cint}([])
        for i in 1:n_linear
            typeof(cones[i]) === ZeroCone{T} ? push!(idx_eq,i) : push!(idx_inq,i) 
        end

        #count up elements and degree
        numel  = sum(cone -> Clarabel.numel(cone), cones; init = 0)
        degree = sum(cone -> Clarabel.degree(cone), cones; init = 0)

        @views numel_linear  = sum(cone -> Clarabel.numel(cone), cones[1:n_linear]; init = 0)
        @views max_linear = maximum(cone -> Clarabel.numel(cone), cones[1:n_linear]; init = 0)
        @views numel_soc  = sum(cone -> Clarabel.numel(cone), cones[n_linear+1:n_linear+n_soc]; init = 0)

        w = CuVector{T}(undef,numel_linear+numel_soc)
        λ = CuVector{T}(undef,numel_linear+numel_soc)
        η = CuVector{T}(undef,n_soc)

        #Initialize space for nonsymmetric cones
        αp = Vector{T}(undef,n_pow)
        pow_ind = n_linear + n_soc + n_exp
        #store the power parameter of each power cone
        for i in 1:n_pow
            αp[i] = cones[i+pow_ind].α
        end

        αp = CuVector(αp)
        H_dual = CuArray{T}(undef,n_exp+n_pow,3,3)
        Hs = CuArray{T}(undef,n_exp+n_pow,3,3)
        grad = CuArray{T}(undef,n_exp+n_pow,3)

        α = CuVector{T}(undef,max(max_linear,n_soc,n_exp,n_pow)) #workspace for step size calculation

        return new(cones,type_counts,numel,degree,CuVector(cpucones.rng_cones),CuVector(cpucones.rng_blocks),_is_symmetric,
                n_linear,idx_eq,idx_inq,
                w,λ,η,
                αp,H_dual,Hs,grad,
                α)
    end
end

CompositeConeGPU(args...) = CompositeConeGPU{DefaultFloat}(args...)


# partial implementation of AbstractArray behaviours
function Base.getindex(S::CompositeConeGPU{T}, i::Int) where {T}
    @boundscheck checkbounds(S.cones,i)
    @inbounds S.cones[i]
end

Base.getindex(S::CompositeConeGPU{T}, b::BitVector) where {T} = S.cones[b]
Base.iterate(S::CompositeConeGPU{T}) where{T} = iterate(S.cones)
Base.iterate(S::CompositeConeGPU{T}, state) where{T} = iterate(S.cones, state)
Base.length(S::CompositeConeGPU{T}) where{T} = length(S.cones)
Base.eachindex(S::CompositeConeGPU{T}) where{T} = eachindex(S.cones)
Base.IndexStyle(S::CompositeConeGPU{T}) where{T} = IndexStyle(S.cones)

function get_type_count(cones::CompositeConeGPU{T}, type::Type) where {T}
    if haskey(cones.type_counts,type)
        return cones.type_counts[type]
    else
        return Cint(0)
    end
end
