using CUDA, CUDA.CUSPARSE

# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------

struct CompositeConeGPU{T} <: AbstractCone{T}
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

    function CompositeConeGPU{T}(cone_specs::Vector{SupportedCone}) where {T}

        #Information from the CompositeCone on CPU 
        n_zero = count(x -> typeof(x) == ZeroConeT, cone_specs)
        n_nn = count(x -> typeof(x) == NonnegativeConeT, cone_specs)
        n_linear = n_zero + n_nn
        n_soc = count(x -> typeof(x) == SecondOrderConeT, cone_specs)
        n_exp = count(x -> typeof(x) == ExponentialConeT, cone_specs)
        n_pow = count(x -> typeof(x) == PowerConeT, cone_specs)
        n_psd = count(x -> typeof(x) == PSDTriangleConeT, cone_specs)

        type_counts = Dict{Type,DefaultInt}()
        if n_zero > 0
            type_counts[ZeroCone] = n_zero
        end
        if n_nn > 0
            type_counts[NonnegativeCone] = n_nn
        end
        if n_soc > 0
            type_counts[SecondOrderCone] = n_soc
        end
        if n_exp > 0
            type_counts[ExponentialCone] = n_exp
        end
        if n_pow > 0
            type_counts[PowerCone] = n_pow
        end
        if n_psd > 0
            type_counts[PSDTriangleCone] = n_psd
        end
        _is_symmetric = (n_exp + n_pow) > 0 ? false : true

        #idx set for eq and ineq constraints
        idx_eq = Vector{Cint}(undef, n_zero)
        idx_inq = Vector{Cint}(undef, n_nn)
        eq_i = zero(Cint)
        inq_i = zero(Cint)
        for i in 1:n_linear
            typeof(cone_specs[i]) === ZeroConeT ? idx_eq[eq_i+=1] = i : idx_inq[inq_i+=1] = i 
        end

        #count up elements and degree
        numel  = sum(cone -> nvars(cone), cone_specs; init = 0)
        degree = sum(cone -> degrees(cone), cone_specs; init = 0)

        #Generate ranges for cones
        rng_cones  = CuVector{UnitRange{Cint}}(collect(rng_cones_iterator(cone_specs)));
        rng_blocks = CuVector{UnitRange{Cint}}(collect(rng_blocks_iterator_full(cone_specs)));

        @views numel_linear  = sum(cone -> nvars(cone), cone_specs[1:n_linear]; init = 0)
        @views numel_soc  = sum(cone -> nvars(cone), cone_specs[n_linear+1:n_linear+n_soc]; init = 0)

        w = CuVector{T}(undef,numel_linear+numel_soc)
        λ = CuVector{T}(undef,numel_linear+numel_soc)
        η = CuVector{T}(undef,n_soc)

        #Initialize space for nonsymmetric cones
        αp = Vector{T}(undef,n_pow)
        pow_ind = n_linear + n_soc + n_exp
        #store the power parameter of each power cone
        for i in 1:n_pow
            αp[i] = cone_specs[i+pow_ind].α
        end

        αp = CuVector(αp)
        H_dual = CuArray{T}(undef,n_exp+n_pow,3,3)
        Hs = CuArray{T}(undef,n_exp+n_pow,3,3)
        grad = CuArray{T}(undef,n_exp+n_pow,3)

        #PSD cone
        #We require all psd cones have the same dimensionality
        psd_ind = pow_ind + n_pow
        psd_dim = (n_psd > 0) ? cone_specs[psd_ind+1].dim : 0
        # for i in 1:n_psd
        #     if(psd_dim != cones[psd_ind+i].n)
        #         throw(DimensionMismatch("Not all positive definite cones have the same dimensionality!"))
        #     end
        # end
        @views @assert(all(cone -> cone.dim == psd_dim, cone_specs[psd_ind+1:psd_ind+n_psd]))

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

        α = CuVector{T}(undef, numel) #workspace for step size calculation and neighborhood check

        return new(type_counts, numel, degree, rng_cones, rng_blocks, _is_symmetric,
                n_linear, n_nn, n_soc, n_exp, n_pow, n_psd,
                idx_eq,idx_inq,
                w,λ,η,
                αp,H_dual,Hs,grad,
                psd_dim, chol1, chol2, SVD, λpsd, Λisqrt, R, Rinv, Hspsd, workmat1, workmat2, workmat3, workvec,
                α)
    end
end

CompositeConeGPU(args...) = CompositeConeGPU{DefaultFloat}(args...)

Base.length(S::CompositeConeGPU{T}) where{T} = length(sum(values(S.type_counts)))

function get_type_count(cones::CompositeConeGPU{T}, type::DataType) where {T}
    typeT = ConeDict[type]
    if haskey(cones.type_counts,typeT)
        return cones.type_counts[typeT]
    else
        return 0
    end
end


# -------------------------------------
# iterators to generate indices into vectors 
# in a cone or cone-related blocks in the Hessian
struct RangeBlocksIteratorFull
    cones::Vector{SupportedCone}
end

function rng_blocks_iterator_full(cones::Vector{SupportedCone})
    RangeBlocksIteratorFull(cones)
end

Base.length(iter::RangeBlocksIteratorFull) = length(iter.cones)

function Base.iterate(iter::RangeBlocksIteratorFull, state=(1, 1)) 
    (coneidx, start) = state 
    if coneidx > length(iter.cones)
        return nothing 
    else 
        cone = iter.cones[coneidx]
        nvars = Clarabel.nvars(cone)
        if (typeof(cone) == ZeroConeT || typeof(cone) == NonnegativeConeT)
            stop = start + nvars - 1
        else
            stop = start + nvars*nvars - 1
        end
        state = (coneidx + 1, stop + 1)
        return (start:stop, state)
    end 
end 
