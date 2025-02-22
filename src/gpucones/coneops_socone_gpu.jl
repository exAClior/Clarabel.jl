# # ----------------------------------------------------
# # Second Order Cone
# # ----------------------------------------------------

# #degree = 1 for SOC, since e'*e = 1
# degree(K::SecondOrderCone{T}) where {T} = 1
# numel(K::SecondOrderCone{T}) where {T} = K.dim

# function is_sparse_expandable(K::SecondOrderCone{T}) where{T}
#     return !isnothing(K.sparse_data)
# end

function _kernel_margins_soc(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views zi = z[rng_cone_i] 
        
        val = zero(T)
        @inbounds for j in 2:size_i 
            val += zi[j]*zi[j]
        end
        α[i]  = zi[1] - sqrt(val)
    end

    return nothing
end

@inline function margins_soc(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint,
    αmin::T
) where{T}
    kernel = @cuda launch=false _kernel_margins_soc(z, α, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(z, α, rng_cones, n_shift, n_soc; threads, blocks)

    @views αsoc = α[1:n_soc]
    αmin = min(αmin,minimum(αsoc))
    CUDA.@sync @. αsoc = max(zero(T),αsoc)
    return (αmin, sum(αsoc))
end

# place vector into socone
function _kernel_scaled_unit_shift_soc!(
    z::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views zi = z[rng_cone_i] 
        zi[1] += α
    end

    return nothing
end

@inline function scaled_unit_shift_soc!(
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    α::T,
    n_shift::Cint,
    n_soc::Cint   
) where{T}

    kernel = @cuda launch=false _kernel_scaled_unit_shift_soc!(z, α, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(z, α, rng_cones, n_shift, n_soc; threads, blocks)
end

# unit initialization for asymmetric solves
function _kernel_unit_initialization_soc!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where{T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        @views zi = z[rng_cone_i] 
        @views si = s[rng_cone_i] 
        zi[1] = one(T)
        @inbounds for j in 2:length(zi)
            zi[j] = zero(T)
        end

        si[1] = one(T)
        @inbounds for j in 2:length(si)
            si[j] = zero(T)
        end
    end
 
    return nothing
end 

@inline function unit_initialization_soc!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}

    kernel = @cuda launch=false _kernel_unit_initialization_soc!(z, s, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(z, s, rng_cones, n_shift, n_soc; threads, blocks)
end 

# # configure cone internals to provide W = I scaling
function _kernel_set_identity_scaling_soc!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views wi = w[rng_cone_i] 
        wi[1] = one(T)
        @inbounds for j in 2:size_i 
            wi[j] = zero(T)
        end
        η[i]  = one(T)

        # YC: no augmented sparse cone at the moment
        # if !isnothing(K.sparse_data)
        #     K.sparse_data.d  = T(0.5)
        #     K.sparse_data.u .= zero(T)
        #     K.sparse_data.u[1] = sqrt(T(0.5))
        #     K.sparse_data.v .= zero(T)
        # end 
    end

    return nothing
end

@inline function set_identity_scaling_soc!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    kernel = @cuda launch=false _kernel_set_identity_scaling_soc!(w, η, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(w, η, rng_cones, n_shift, n_soc; threads, blocks)
end

function _kernel_update_scaling_soc!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views zi = z[rng_i] 
        @views si = s[rng_i] 
        @views wi = w[rng_i] 
        @views λi = λ[rng_i]

        #first calculate the scaled vector w
        @views zscale = _sqrt_soc_residual_gpu(zi)
        @views sscale = _sqrt_soc_residual_gpu(si)

        #the leading scalar term for W^TW
        η[i] = sqrt(sscale/zscale)

        # construct w and normalize
        @inbounds for k in rng_i
            w[k] = s[k]/(sscale)
        end

        wi[1]  += zi[1]/(zscale)

        @inbounds for j in 2:length(wi)
            wi[j] -= zi[j]/(zscale)
        end
    
        wscale = _sqrt_soc_residual_gpu(wi)
        wi ./= wscale

        #try to force badly scaled w to come out normalized
        w1sq = zero(T)
        @inbounds for j in 2:length(wi)
            w1sq += wi[j]*wi[j]
        end
        wi[1] = sqrt(1 + w1sq)

        #Compute the scaling point λ.   Should satisfy λ = Wz = W^{-T}s
        γi = 0.5 * wscale
        λi[1] = γi 

        coef = inv(si[1]/sscale + zi[1]/zscale + 2*γi)
        c1 = ((γi + zi[1]/zscale)/sscale)
        c2 = ((γi + si[1]/sscale)/zscale)
        @inbounds for j in 2:length(λi)
            λi[j] = coef*(c1*si[j] +c2*zi[j])
        end
        λi .*= sqrt(sscale*zscale)
    end

    return nothing

    #YC: don't support sparse SOC currently
    # #Populate sparse expansion terms if allocated
    # if is_sparse_expandable(K)

    #     sparse_data = K.sparse_data

    #     #various intermediate calcs for u,v,d
    #     α  = 2*w[1]

    #     #Scalar d is the upper LH corner of the diagonal
    #     #term in the rank-2 update form of W^TW
    #     wsq    = w[1]*w[1] + w1sq
    #     wsqinv = 1/wsq
    #     sparse_data.d    = wsqinv / 2

    #     #the vectors for the rank two update
    #     #representation of W^TW
    #     u0  = sqrt(wsq - sparse_data.d)
    #     u1 = α/u0
    #     v0 = zero(T)
    #     v1 = sqrt( 2*(2 + wsqinv)/(2*wsq - wsqinv))
        
    #     sparse_data.u[1] = u0
    #     @views K.sparse_data.u[2:end] .= u1.*w[2:end]
    #     sparse_data.v[1] = v0
    #     @views sparse_data.v[2:end] .= v1.*w[2:end]

    # end 

    # return is_scaling_success = true
end

@inline function update_scaling_soc!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_update_scaling_soc!(s, z, w, λ, η, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(s, z, w, λ, η, rng_cones, n_shift, n_soc; threads, blocks)
end

function _kernel_get_Hs_soc!(
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        rng_block_i = rng_blocks[shift_i]
        size_i = length(rng_cone_i)
        @views wi = w[rng_cone_i] 
        @views Hsblocki = Hsblocks[rng_block_i]

        hidx = one(Cint)
        @inbounds for col in rng_cone_i
            wcol = w[col]
            @inbounds for row in rng_cone_i
                Hsblocki[hidx] = 2*w[row]*wcol
                hidx += 1
            end 
        end
        Hsblocki[1] -= one(T)
        @inbounds for ind in 2:size_i
            Hsblocki[(ind-1)*size_i + ind] += one(T)
        end
        Hsblocki .*= η[i]^2
    end

    return nothing
end

@inline function get_Hs_soc!(
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    #YC: don't support sparse SOC currently
    # if is_sparse_expandable(K)
    #     #For sparse form, we are returning here the diagonal D block 
    #     #from the sparse representation of W^TW, but not the
    #     #extra two entries at the bottom right of the block.
    #     #The AbstractVector for s and z (and its views) don't
    #     #know anything about the 2 extra sparsifying entries
    #     Hsblock    .= K.η^2
    #     Hsblock[1] *= K.sparse_data.d
    # end

    kernel = @cuda launch=false _kernel_get_Hs_soc!(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_soc; threads, blocks)

end

# function Hs_is_diagonal(
#     K::SecondOrderCone{T}
# ) where{T}
#     return is_sparse_expandable(K)
# end

# compute the product y = WᵀWx
function _kernel_mul_Hs_soc!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    # y = = H^{-1}x = W^TWx
    # where H^{-1} = \eta^{2} (2*ww^T - J)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views xi = x[rng_cone_i] 
        @views yi = y[rng_cone_i] 
        @views wi = w[rng_cone_i] 

        c = 2*_dot_xy_gpu(wi,xi,1:size_i)

        yi[1] = -xi[1] + c*wi[1]
        @inbounds for j in 2:size_i
            yi[j] = xi[j] + c*wi[j]
        end

        _multiply_gpu(yi,η[i]^2)
    end

    return nothing
end

@inline function mul_Hs_soc!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_mul_Hs_soc!(y, x, w, η, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(y, x, w, η, rng_cones, n_shift, n_soc; threads, blocks)
end

# returns x = λ ∘ λ for the socone
function _kernel_affine_ds_soc!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views dsi = ds[rng_cone_i] 
        @views λi = λ[rng_cone_i] 

        #circ product λ∘λ
        dsi[1] = zero(T)
        for j in 1:length(dsi)
            dsi[1] += λi[j]*λi[j]
        end
        λi0 = λi[1]
        for j = 2:length(dsi)
            dsi[j] = 2*λi0*λi[j]
        end
      
    end

    return nothing

end

@inline function affine_ds_soc!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_affine_ds_soc!(ds, λ, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(ds, λ, rng_cones, n_shift, n_soc; threads, blocks)
end

function _kernel_combined_ds_shift_soc!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    σμ::T
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views step_zi = step_z[rng_cone_i] 
        @views step_si = step_s[rng_cone_i] 
        @views wi = w[rng_cone_i] 
        @views shifti = shift[rng_cone_i] 
    
        #shift vector used as workspace for a few steps 
        tmp = shifti            

        #Δz <- WΔz
        @inbounds for j in 1:size_i
            tmp[j] = step_zi[j]
        end         
        ζ = zero(T)
        
        @inbounds for j in 2:size_i
            ζ += wi[j]*tmp[j]
        end

        c = tmp[1] + ζ/(1+wi[1])
      
        step_zi[1] = η[i]*(wi[1]*tmp[1] + ζ)
      
        @inbounds for j in 2:size_i
            step_zi[j] = η[i]*(tmp[j] + c*wi[j]) 
        end      

        #Δs <- W⁻¹Δs
        @inbounds for j in 1:size_i
            tmp[j] = step_si[j]
        end           
        ζ = zero(T)
        @inbounds for j in 2:size_i
            ζ += wi[j]*tmp[j]
        end

        c = -tmp[1] + ζ/(1+wi[1])
    
        step_si[1] = (one(T)/η[i])*(wi[1]*tmp[1] - ζ)
    
        @inbounds for j = 2:size_i
            step_si[j] = (one(T)/η[i])*(tmp[j] + c*wi[j])
        end

        #shift = W⁻¹Δs ∘ WΔz - σμe  
        val = zero(T)
        @inbounds for j in 1:size_i
            val += step_si[j]*step_zi[j]
        end       
        shifti[1] = val - σμ 

        s0   = step_si[1]
        z0   = step_zi[1]
        for j = 2:size_i
            shifti[j] = s0*step_zi[j] + z0*step_si[j]
        end      
    end                    

    return nothing
end

@inline function combined_ds_shift_soc!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint,
    σμ::T
) where {T}

    kernel = @cuda launch=false _kernel_combined_ds_shift_soc!(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ; threads, blocks)
end

function _kernel_Δs_from_Δz_offset_soc!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views outi = out[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views wi = w[rng_cone_i] 
        @views λi = λ[rng_cone_i] 

        #out = Wᵀ(λ \ ds).  Below is equivalent,
        #but appears to be a little more stable 
        reszi = _soc_residual_gpu(zi)

        @views λ1ds1  = _dot_xy_gpu(λi,dsi,2:size_i)
        @views w1ds1  = _dot_xy_gpu(wi,dsi,2:size_i)

        _minus_vec_gpu(outi,zi)
        outi[1] = +zi[1]
    
        c = λi[1]*dsi[1] - λ1ds1
        _multiply_gpu(outi,c/reszi)

        outi[1] += η[i]*w1ds1
        @inbounds for j in 2:size_i
            outi[j] += η[i]*(dsi[j] + w1ds1/(1+wi[1])*wi[j])
        end
    
        _multiply_gpu(outi,one(T)/λi[1])
    end

    return nothing

end

@inline function Δs_from_Δz_offset_soc!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_Δs_from_Δz_offset_soc!(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc; threads, blocks)
end

#return maximum allowable step length while remaining in the socone
function _kernel_step_length_soc(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     rng_cones::AbstractVector,
     n_linear::Cint,
     n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        @views si = s[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views dzi = dz[rng_cone_i]         

        αz   = _step_length_soc_component_gpu(zi,dzi,α[i])
        αs   = _step_length_soc_component_gpu(si,dsi,α[i])
        α[i] = min(αz,αs)
    end

    return nothing
end

@inline function step_length_soc(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αmax::T,
     rng_cones::AbstractVector,
     n_shift::Cint,
     n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_step_length_soc(dz, ds, z, s, α, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(dz, ds, z, s, α, rng_cones, n_shift, n_soc; threads, blocks)
    @views αmax = min(αmax,minimum(α[1:n_soc]))

    return αmax
end

function _kernel_compute_barrier_soc(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        @views si = s[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views dzi = dz[rng_cone_i]  
        res_si = _soc_residual_shifted(si,dsi,α)
        res_zi = _soc_residual_shifted(zi,dzi,α)

        # avoid numerical issue if res_s <= 0 or res_z <= 0
        barrier[i] = (res_si > 0 && res_zi > 0) ? -logsafe(res_si*res_zi)/2 : Inf
    end

    return nothing
end

@inline function compute_barrier_soc(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_compute_barrier_soc(barrier,z,s,dz,ds,α,rng_cones,n_linear,n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(barrier,z,s,dz,ds,α,rng_cones,n_linear,n_soc; threads, blocks)

    return sum(barrier[1:n_soc])
end

# # ---------------------------------------------
# # operations supported by symmetric cones only 
# # ---------------------------------------------


# # implements y = αWx + βy for the socone
# function mul_W!(
#     K::SecondOrderCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#   #NB: symmetric, so ignore transpose

#   # use the fast product method from ECOS ECC paper
#   @views ζ = dot(K.w[2:end],x[2:end])
#   c = x[1] + ζ/(1+K.w[1])

#   y[1] = α*K.η*(K.w[1]*x[1] + ζ) + β*y[1]

#   @inbounds for i in 2:length(y)
#       y[i] = α*K.η*(x[i] + c*K.w[i]) + β*y[i]
#   end

#   return nothing
# end

# # implements y = αW^{-1}x + βy for the socone
# function mul_Winv!(
#     K::SecondOrderCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#     #NB: symmetric, so ignore transpose

#     # use the fast inverse product method from ECOS ECC paper
#     @views ζ = dot(K.w[2:end],x[2:end])
#     c = -x[1] + ζ/(1+K.w[1])

#     y[1] = (α/K.η)*(K.w[1]*x[1] - ζ) + β*y[1]

#     @inbounds for i = 2:length(y)
#         y[i] = (α/K.η)*(x[i] + c*K.w[i]) + β*y[i]
#     end

#     return nothing
# end

# # implements x = λ \ z for the socone, where λ
# # is the internally maintained scaling variable.
# function λ_inv_circ_op!(
#     K::SecondOrderCone{T},
#     x::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     inv_circ_op!(K, x, K.λ, z)

# end

# ---------------------------------------------
# Jordan algebra operations for symmetric cones 
# ---------------------------------------------

# # implements x = y \ z for the socone
# function inv_circ_op!(
#     K::SecondOrderCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     p = _soc_residual(y)
#     pinv = 1/p
#     @views v = dot(y[2:end],z[2:end])

#     x[1]      = (y[1]*z[1] - v)*pinv
#     @views x[2:end] .= pinv.*(v/y[1] - z[1]).*y[2:end] + (1/y[1]).*z[2:end]

#     return nothing
# end

# ---------------------------------------------
# internal operations for second order cones 
# ---------------------------------------------

@inline function _soc_residual_gpu(z::AbstractVector{T}) where {T} 
    res = z[1]*z[1]
    for j in 2:length(z)
        res -= z[j]*z[j]
    end
    
    return res
end 

@inline function _sqrt_soc_residual_gpu(z::AbstractVector{T}) where {T} 
    res = _soc_residual_gpu(z)
    
    # set res to 0 when z is not an interior point
    res = res > 0.0 ? sqrt(res) : zero(T)
end 

@inline function _dot_xy_gpu(x::AbstractVector{T},y::AbstractVector{T},rng::UnitRange) where {T} 
    val = zero(T)
    for j in rng
        val += x[j]*y[j]
    end
    
    return val
end 

@inline function _minus_vec_gpu(y::AbstractVector{T},x::AbstractVector{T}) where {T} 
    @inbounds for j in 1:length(x)
        y[j] = -x[j]
    end
end 

@inline function _multiply_gpu(x::AbstractVector{T},a::T) where {T} 
    @inbounds for j in 1:length(x)
        x[j] *= a 
    end
end 

# find the maximum step length α≥0 so that
# x + αy stays in the SOC
@inline function _step_length_soc_component_gpu(
    x::AbstractVector{T},
    y::AbstractVector{T},
    αmax::T
) where {T}

    # assume that x is in the SOC, and find the minimum positive root
    # of the quadratic equation:  ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    @views a = _soc_residual_gpu(y) #NB: could be negative
    @views b = 2*(x[1]*y[1] - _dot_xy_gpu(x,y,2:length(x)))
    @views c = max(zero(T),_soc_residual_gpu(x)) #should be ≥0
    d = b^2 - 4*a*c

    if(c < 0)
        # This should never be reachable since c ≥ 0 above
        return -Inf
    end

    if( (a > 0 && b > 0) || d < 0)
        #all negative roots / complex root pair
        #-> infinite step length
        return αmax

    elseif a == 0
        #edge case with only one root.  This corresponds to
        #the case where the search direction is exactly on the 
        #cone boundary.   The root should be -c/b, but b can't 
        #be negative since both (x,y) are in the cone and it is 
        #self dual, so <x,y> \ge 0 necessarily.
        return αmax

    elseif c == 0
        #Edge case with one of the roots at 0.   This corresponds 
        #to the case where the initial point is exactly on the 
        #cone boundary.  The other root is -b/a.   If the search 
        #direction is in the cone, then a >= 0 and b can't be 
        #negative due to self-duality.  If a < 0, then the 
        #direction is outside the cone and b can't be positive.
        #Either way, step length is determined by whether or not 
        #the search direction is in the cone.

        return (a >= 0 ? αmax : zero(T)) 
    end 


    # if we got this far then we need to calculate a pair 
    # of real roots and choose the smallest positive one.  
    # We need to be cautious about cancellations though.  
    # See §1.4: Goldberg, ACM Computing Surveys, 1991 
    # https://dl.acm.org/doi/pdf/10.1145/103162.103163

    t = (b >= 0) ? (-b - sqrt(d)) : (-b + sqrt(d))

    r1 = (2*c)/t;
    r2 = t/(2*a);

    #return the minimum positive root, up to αmax
    r1 = r1 < 0 ? floatmax(T) : r1
    r2 = r2 < 0 ? floatmax(T) : r2

    return min(αmax,r1,r2)

end