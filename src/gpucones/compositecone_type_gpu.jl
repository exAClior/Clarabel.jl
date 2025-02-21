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
    n_nn::Cint
    n_soc::Cint
    n_exp::Cint
    n_pow::Cint
    n_psd::Cint

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

    #PSD cone
    psd_dim::Cint                  #We only support PSD cones with the same small dimension
    chol1::AbstractArray{T,3}
    chol2::AbstractArray{T,3}
    SVD::AbstractArray{T,3}
    λpsd::AbstractMatrix{T}
    Λisqrt::AbstractMatrix{T}
    R::AbstractArray{T,3}
    Rinv::AbstractArray{T,3}
    Hspsd::AbstractArray{T,3}

    #workspace for various internal uses
    workmat1::AbstractArray{T,3}
    workmat2::AbstractArray{T,3}
    workmat3::AbstractArray{T,3}
    workvec::AbstractVector{T}

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
        n_psd = haskey(type_counts,PSDTriangleCone) ? type_counts[PSDTriangleCone] : 0

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

        #PSD cone
        #We require all psd cones have the same dimensionality
        psd_ind = pow_ind + n_pow
        psd_dim = haskey(type_counts,PSDTriangleCone) ? cones[psd_ind+1].n : 0
        for i in 1:n_psd
            if(psd_dim != cones[psd_ind+i].n)
                throw(DimensionMismatch("Not all positive definite cones have the same dimensionality!"))
            end
        end

        chol1 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        chol2 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        SVD   = CUDA.zeros(T,psd_dim,psd_dim,n_psd)

        λpsd   = CUDA.zeros(T,psd_dim,n_psd)
        Λisqrt = CUDA.zeros(T,psd_dim,n_psd)
        R      = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        Rinv   = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        Hspsd  = CUDA.zeros(T,triangular_number(psd_dim),triangular_number(psd_dim),n_psd)

        workmat1 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        workmat2 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        workmat3 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        workvec  = CUDA.zeros(T,triangular_number(psd_dim)*n_psd)

        α = CuVector{T}(undef,sum(cone -> Clarabel.numel(cone), cones; init = 0)) #workspace for step size calculation and neighborhood check

        return new(cones,type_counts,numel,degree,CuVector(cpucones.rng_cones),CuVector(cpucones.rng_blocks),_is_symmetric,
                n_linear, n_nn, n_soc, n_exp, n_pow, n_psd,
                idx_eq,idx_inq,
                w,λ,η,
                αp,H_dual,Hs,grad,
                psd_dim,chol1,chol2,SVD,λpsd,Λisqrt,R,Rinv,Hspsd,workmat1,workmat2,workmat3,workvec,
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
