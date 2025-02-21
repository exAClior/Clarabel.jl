# ## ------------------------------------
# # Nonnegative Cone
# # -------------------------------------

# degree(K::NonnegativeCone{T}) where {T} = K.dim
# numel(K::NonnegativeCone{T}) where {T} = K.dim

# function rectify_equilibration!(
#     K::NonnegativeCone{T},
#     δ::AbstractVector{T},
#     e::AbstractVector{T}
# ) where{T}

#     #allow elementwise equilibration scaling
#     δ .= one(T)
#     return false
# end

function margins_nonnegative(
    z::AbstractVector{T},
    α::AbstractVector{T}
) where{T}

    @. α = max(z,0)
    CUDA.synchronize()

    return nothing
end

# place vector into nn cone
function scaled_unit_shift_nonnegative!(
    z::AbstractVector{T},
    α::T
) where{T}

    @. z += α 

    return nothing
end

# unit initialization for asymmetric solves
function unit_initialization_nonnegative!(
   z::AbstractVector{T},
   s::AbstractVector{T}
) where{T}

    s .= one(T)
    z .= one(T)

   return nothing
end

#configure cone internals to provide W = I scaling
function set_identity_scaling_nonnegative!(
    w::AbstractVector{T}
) where {T}

    @. w = one(T)

    return nothing
end

function update_scaling_nonnegative!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T}
) where {T}

    @. λ = sqrt(s*z)
    @. w = sqrt(s/z)

    return is_scaling_success = true
end

function get_Hs_nonnegative!(
    Hsblock::AbstractVector{T},
    w::AbstractVector{T}
) where {T}

    #this block is diagonal, and we expect here
    #to receive only the diagonal elements to fill
    @. Hsblock = w^2

    return nothing
end

# compute the product y = WᵀWx
function mul_Hs_nonnegative!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T}
) where {T}

    #NB : seemingly sensitive to order of multiplication
    @. y = (w * (w * x))

end

# returns ds = λ∘λ for the nn cone
function affine_ds_nonnegative!(
    ds::AbstractVector{T},
    λ::AbstractVector{T}
) where {T}

    @. ds = λ^2

    return nothing
end

@inline function combined_ds_shift_nonnegative!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    σμ::T
) where {T}

    # The shift must be assembled carefully if we want to be economical with
    # allocated memory.  Will modify the step.z and step.s in place since
    # they are from the affine step and not needed anymore.
    #
    # We can't have aliasing vector arguments to gemv_W or gemv_Winv, so 
    # we need a temporary variable to assign #Δz <= WΔz and Δs <= W⁻¹Δs

    #shift vector used as workspace for a few steps 
    tmp = shift              

     #Δz <- WΔz
    @. tmp = step_z           
    @. step_z = tmp*w

    #Δs <- W⁻¹Δs
    @. tmp = step_s           
    @. step_s = tmp/w

    #shift = W⁻¹Δs ∘ WΔz - σμe
    @. shift = step_s*step_z - σμ                       

    return nothing
end

function Δs_from_Δz_offset_nonnegative!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T}
) where {T}
    @. out = ds / z
end

#return maximum allowable step length while remaining in the nn cone
function _kernel_step_length_nonnegative(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     len_rng::Cint,
     αmax::T
) where {T}


    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= len_rng
        αz = dz[i] < 0 ? (min(αmax,-z[i]/dz[i])) : αmax
        αs = ds[i] < 0 ? (min(αmax,-s[i]/ds[i])) : αmax
        α[i] = min(αz, αs)
    end

    return nothing
end

@inline function step_length_nonnegative(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     len_nn::Cint,
     αmax::T
) where {T}

    kernel = @cuda launch=false _kernel_step_length_nonnegative(dz,ds,z,s,α,len_nn,αmax)
    config = launch_configuration(kernel.fun)
    threads = min(len_nn, config.threads)
    blocks = cld(len_nn, threads)

    CUDA.@sync kernel(dz,ds,z,s,α,len_nn,αmax; threads, blocks)

    return minimum(α)
end

function _kernel_compute_barrier_nonnegative(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    len_nn::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= len_nn
        barrier[i] = -logsafe((s[i] + α*ds[i])*(z[i] + α*dz[i]))
    end

    return nothing
end

@inline function compute_barrier_nonnegative(
    work::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    len_nn::Cint
) where {T}

    barrier = zero(T)
    CUDA.@allowscalar for i in idx_inq
        rng_cone_i = rng_cones[i]
        @views worki = work[rng_cone_i]
        @views @. worki = -logsafe((s[rng_cone_i] + α*ds[rng_cone_i])*(z[rng_cone_i] + α*dz[rng_cone_i]))
        CUDA.synchronize()
        barrier += sum(worki)
    end

    return barrier
end

# # ---------------------------------------------
# # operations supported by symmetric cones only 
# # ---------------------------------------------

# # implements y = αWx + βy for the nn cone
# function mul_W!(
#     K::NonnegativeCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#   #W is diagonal so ignore transposition
#   #@. y = α*(x*K.w) + β*y
#   @inbounds for i = eachindex(y)
#       y[i] = α*(x[i]*K.w[i]) + β*y[i]
#   end

#   return nothing
# end

# # implements y = αW^{-1}x + βy for the nn cone
# function mul_Winv!(
#     K::NonnegativeCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#   #W is diagonal, so ignore transposition
#   #@. y = α*(x/K.w) + β.*y
#   @inbounds for i = eachindex(y)
#       y[i] = α*(x[i]/K.w[i]) + β*y[i]
#   end

#   return nothing
# end

# # implements x = λ \ z for the nn cone, where λ
# # is the internally maintained scaling variable.
# function λ_inv_circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     inv_circ_op!(K, x, K.λ, z)

# end

# # ---------------------------------------------
# # Jordan algebra operations for symmetric cones 
# # ---------------------------------------------

# # implements x = y ∘ z for the nn cone
# function circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     @. x = y*z

#     return nothing
# end

# # implements x = y \ z for the nn cone
# function inv_circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     @. x = z/y

#     return nothing
# end