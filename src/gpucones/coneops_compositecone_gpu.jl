
degree(cones::CompositeConeGPU{T}) where {T} = cones.degree
numel(cones::CompositeConeGPU{T}) where {T}  = cones.numel

# # -----------------------------------------------------
# # dispatch operators for multiple cones
# # -----------------------------------------------------

# function is_symmetric(cones::CompositeCone{T}) where {T}
#     #true if all pieces are symmetric.  
#     #determined during obj construction
#     return cones._is_symmetric
# end

# function is_sparse_expandable(cones::CompositeCone{T}) where {T}
    
#     #This should probably never be called
#     #any(is_sparse_expandable, cones)
#     ErrorException("This function should not be reachable")
    
# end

# function allows_primal_dual_scaling(cones::CompositeCone{T}) where {T}
#     all(allows_primal_dual_scaling, cones)
# end


# function rectify_equilibration!(
#     cones::CompositeCone{T},
#      δ::ConicVector{T},
#      e::ConicVector{T}
# ) where{T}

#     any_changed = false

#     #we will update e <- δ .* e using return values
#     #from this function.  default is to do nothing at all
#     δ .= 1

#     for (cone,δi,ei) in zip(cones,δ.views,e.views)
#         @conedispatch any_changed |= rectify_equilibration!(cone,δi,ei)
#     end

#     return any_changed
# end

function margins(
    cones::CompositeConeGPU{T},
    z::AbstractVector{T},
    pd::PrimalOrDualCone,
) where {T}
    αmin = typemax(T)
    β = zero(T)

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    rng_cones = cones.rng_cones
    type_counts = cones.type_counts
    α = cones.α
    @. α = αmin
    
    CUDA.@allowscalar for i in type_counts[NonnegativeCone]
        rng_cone_i = rng_cones[i]
        @views zi = z[rng_cone_i]
        len = length(zi)
        αmin = min(αmin,minimum(zi))
        @views αnn = α[1:len]
        margins_nonnegative(zi,αnn)
        β += sum(αnn)
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_margins_soc(z,α,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(z,α,rng_cones,n_linear,n_soc; threads, blocks)

        @views αsoc = α[1:n_soc]
        αmin = min(αmin,minimum(αsoc))
        CUDA.@sync @. αsoc = max(zero(T),αsoc)
        β += sum(αsoc)
    end

    return (αmin,β)
end

function scaled_unit_shift!(
    cones::CompositeConeGPU{T},
    z::AbstractVector{T},
    α::T,
    pd::PrimalOrDualCone
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    rng_cones = cones.rng_cones
    type_counts = cones.type_counts

    CUDA.@allowscalar begin
        for i in type_counts[ZeroCone]
            rng_cone_i = rng_cones[i]
            @views scaled_unit_shift_zero!(z[rng_cone_i],pd)
        end
        for i in type_counts[NonnegativeCone]
            rng_cone_i = rng_cones[i]
            @views scaled_unit_shift_nonnegative!(z[rng_cone_i],α)
        end
        CUDA.synchronize()
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_scaled_unit_shift_soc!(z,α,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(z,α,rng_cones,n_linear,n_soc; threads, blocks)
    end

    return nothing

    # for (cone,zi) in zip(cones,z.views)
    #     @conedispatch scaled_unit_shift!(cone,zi,α,pd)
    # end

    # return nothing
end

# # unit initialization for asymmetric solves
# function unit_initialization!(
#     cones::CompositeCone{T},
#     z::ConicVector{T},
#     s::ConicVector{T}
# ) where {T}

#     for (cone,zi,si) in zip(cones,z.views,s.views)
#         @conedispatch unit_initialization!(cone,zi,si)
#     end
#     return nothing
# end

function set_identity_scaling!(
    cones::CompositeConeGPU{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    rng_cones = cones.rng_cones
    type_counts = cones.type_counts
    w = cones.w
    η = cones.η
    
    CUDA.@allowscalar for i in type_counts[NonnegativeCone]
        @views set_identity_scaling_nonnegative!(w[rng_cones[i]])
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_set_identity_scaling_soc!(w,η,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(w,η,rng_cones,n_linear,n_soc; threads, blocks)
    end

    return nothing
end

function update_scaling!(
    cones::CompositeConeGPU{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    rng_cones = cones.rng_cones
    type_counts = cones.type_counts
    w = cones.w
    λ = cones.λ
    η = cones.η
    is_scaling_success = true
    
    CUDA.@allowscalar for i in type_counts[NonnegativeCone]
        @views is_scaling_success = update_scaling_nonnegative!(s[rng_cones[i]],z[rng_cones[i]],w[rng_cones[i]],λ[rng_cones[i]])

        if !is_scaling_success
            return is_scaling_success = false
        end
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_update_scaling_soc!(s,z,w,λ,η,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(s,z,w,λ,η,rng_cones,n_linear,n_soc; threads, blocks)
    end

    return is_scaling_success = true
end

# The Hs block for each cone.
function get_Hs!(
    cones::CompositeConeGPU{T},
    Hsblocks::AbstractVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    type_counts = cones.type_counts
    rng_blocks = cones.rng_blocks
    rng_cones = cones.rng_cones
    w = cones.w

    CUDA.@allowscalar begin
        for i in type_counts[ZeroCone]
            @views get_Hs_zero!(Hsblocks[rng_blocks[i]])
        end
        for i in type_counts[NonnegativeCone]
            @views get_Hs_nonnegative!(Hsblocks[rng_blocks[i]],w[rng_cones[i]])
        end
        CUDA.synchronize()
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_get_Hs_soc!(Hsblocks,cones.w,cones.η,rng_cones,rng_blocks,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(Hsblocks,cones.w,cones.η,rng_cones,rng_blocks,n_linear,n_soc; threads, blocks)
    end

    return nothing
end

# compute the generalized product :
# WᵀWx for symmetric cones 
# μH(s)x for symmetric cones

function mul_Hs!(
    cones::CompositeConeGPU{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    type_counts = cones.type_counts
    rng_cones = cones.rng_cones
    w = cones.w
    η = cones.η

    CUDA.@allowscalar begin
        for i in type_counts[ZeroCone]
            @views mul_Hs_zero!(y[rng_cones[i]])
        end
        for i in type_counts[NonnegativeCone]
            @views mul_Hs_nonnegative!(y[rng_cones[i]],x[rng_cones[i]],w[rng_cones[i]])
        end
        CUDA.synchronize()
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_mul_Hs_soc!(y,x,w,η,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(y,x,w,η,rng_cones,n_linear,n_soc; threads, blocks)
    end

    return nothing
end

# x = λ ∘ λ for symmetric cone and x = s for asymmetric cones
function affine_ds!(
    cones::CompositeConeGPU{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    type_counts = cones.type_counts
    rng_cones = cones.rng_cones
    λ = cones.λ

    CUDA.@allowscalar begin
        for i in type_counts[ZeroCone]
            @views affine_ds_zero!(ds[rng_cones[i]])
        end
        for i in type_counts[NonnegativeCone]
            @views affine_ds_nonnegative!(ds[rng_cones[i]],λ[rng_cones[i]])
        end
        CUDA.synchronize()
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_affine_ds_soc!(ds,cones.λ,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(ds,cones.λ,rng_cones,n_linear,n_soc; threads, blocks)
    end

    return nothing
end

function combined_ds_shift!(
    cones::CompositeConeGPU{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    type_counts = cones.type_counts
    rng_cones = cones.rng_cones
    w = cones.w
    η = cones.η

    CUDA.@allowscalar begin
        for i in type_counts[ZeroCone]
            @views combined_ds_shift_zero!(shift[rng_cones[i]])
        end
        for i in type_counts[NonnegativeCone]
            @views combined_ds_shift_nonnegative!(shift[rng_cones[i]],step_z[rng_cones[i]],step_s[rng_cones[i]],w[rng_cones[i]],σμ)
        end
        CUDA.synchronize()
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_combined_ds_shift_soc!(shift,step_z,step_s,w,η,rng_cones,n_linear,n_soc,σμ)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(shift,step_z,step_s,w,η,rng_cones,n_linear,n_soc,σμ; threads, blocks)
    end

    # return nothing

end

function Δs_from_Δz_offset!(
    cones::CompositeConeGPU{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    type_counts = cones.type_counts
    rng_cones = cones.rng_cones
    w = cones.w
    λ = cones.λ
    η = cones.η

    CUDA.@allowscalar begin
        for i in type_counts[ZeroCone]
            @views Δs_from_Δz_offset_zero!(out[rng_cones[i]])
        end
        for i in type_counts[NonnegativeCone]
            @views Δs_from_Δz_offset_nonnegative!(out[rng_cones[i]],ds[rng_cones[i]],z[rng_cones[i]])
        end
        CUDA.synchronize()
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_Δs_from_Δz_offset_soc!(out,ds,z,w,λ,η,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(out,ds,z,w,λ,η,rng_cones,n_linear,n_soc; threads, blocks)
    end
    return nothing
end

# maximum allowed step length over all cones
function step_length(
     cones::CompositeConeGPU{T},
        dz::AbstractVector{T},
        ds::AbstractVector{T},
         z::AbstractVector{T},
         s::AbstractVector{T},
  settings::Settings{T},
      αmax::T,
) where {T}

    n_linear        = cones.n_linear
    n_soc           = cones.n_soc
    type_counts     = cones.type_counts
    rng_cones       = cones.rng_cones
    α               = cones.α

    CUDA.@sync @. α = αmax          #Initialize step size

    CUDA.@allowscalar begin
        for i in type_counts[NonnegativeCone]
            len_nn = Cint(length(rng_cones[i]))
            rng_cone_i = rng_cones[i]
            kernel = @cuda launch=false _kernel_step_length_nonnegative!(dz,ds,z,s,α,rng_cone_i)
            config = launch_configuration(kernel.fun)
            threads = min(len_nn, config.threads)
            blocks = cld(len_nn, threads)
    
            CUDA.@sync kernel(dz,ds,z,s,α,rng_cone_i; threads, blocks)
            αmax = minimum(α)
        end
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_step_length_soc!(dz,ds,z,s,α,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(dz,ds,z,s,α,rng_cones,n_linear,n_soc; threads, blocks)
        αmax = min(αmax,minimum(α))
        if αmax < 0
            throw(DomainError("starting point of line search not in SOC"))
        end
    end

    return (αmax,αmax)
end

# # compute the total barrier function at the point (z + α⋅dz, s + α⋅ds)
# function compute_barrier(
#     cones::CompositeCone{T},
#     z::ConicVector{T},
#     s::ConicVector{T},
#     dz::ConicVector{T},
#     ds::ConicVector{T},
#     α::T
# ) where {T}

#     dz    = dz.views
#     ds    = ds.views
#     z     = z.views
#     s     = s.views

#     barrier = zero(T)

#     for (cone,zi,si,dzi,dsi) in zip(cones,z,s,dz,ds)
#         @conedispatch barrier += compute_barrier(cone,zi,si,dzi,dsi,α)
#     end

#     return barrier
# end

