using CUDA, CUDA.CUSPARSE

# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------
@enum ConeType begin
    ZEROCONE           = zero(Cint)
    NONNEGATIVECONE
    SECONDORDERCONE
    EXPONENTIALCONE
    POWERCONE
end

const ConeTypeDict = Dict{DataType,Cint}(
           ZeroConeT => 0,
    NonnegativeConeT => 1,
    SecondOrderConeT => 2,
    ExponentialConeT => 3,
          PowerConeT => 4
)

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

    function CompositeConeGPU{T}(cone_specs::Vector{SupportedCone}) where {T}

        ncones = length(cone_specs)
        cones  = AbstractCone{T}[]
        sizehint!(cones,ncones)

        type_counts = Dict{Type,Cint}()

        #assumed symmetric to start
        _is_symmetric = true

        #create cones with the given dims
        for coneT in cone_specs
            #make a new cone
            cone = make_cone(T, coneT);

            #update global problem symmetry
            _is_symmetric = _is_symmetric && is_symmetric(cone)

            #increment type counts 
            key = ConeDict[typeof(coneT)]
            haskey(type_counts,key) ? type_counts[key] += 1 : type_counts[key] = 1
            
            push!(cones,cone)
        end

        n_linear = type_counts[ZeroCone] + type_counts[NonnegativeCone]
        n_soc = haskey(type_counts,SecondOrderCone) ? type_counts[SecondOrderCone] : 0
        n_exp = haskey(type_counts,ExponentialCone) ? type_counts[ExponentialCone] : 0
        n_pow = haskey(type_counts,PowerCone) ? type_counts[PowerCone] : 0
        #idx set for eq and ineq constraints
        idx_eq = Vector{Cint}([])
        idx_inq = Vector{Cint}([])
        for i in 1:n_linear
            typeof(cone_specs[i]) == ZeroConeT ? push!(idx_eq,i) : push!(idx_inq,i) 
        end

        #count up elements and degree
        numel  = sum(cone -> Clarabel.numel(cone), cones; init = 0)
        degree = sum(cone -> Clarabel.degree(cone), cones; init = 0)

        #ranges for the subvectors associated with each cone,
        #and the range for the corresponding entries
        #in the Hs sparse block

        rng_cones_cpu  = collect(UnitRange{Cint},rng_cones_iterator(cones));
        rng_blocks_cpu = collect(UnitRange{Cint},rng_blocks_iterator(cones,true));

        numel_linear  = sum(cone -> Clarabel.numel(cone), cones[1:n_linear]; init = 0)
        max_linear = maximum(cone -> Clarabel.numel(cone), cones[1:n_linear]; init = 0)
        numel_soc  = sum(cone -> Clarabel.numel(cone), cones[n_linear+1:n_linear+n_soc]; init = 0)

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

        return new(cones,type_counts,numel,degree,CuVector(rng_cones_cpu),CuVector(rng_blocks_cpu),_is_symmetric,
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
