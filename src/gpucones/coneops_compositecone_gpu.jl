
degree(cones::CompositeConeGPU{T}) where {T} = cones.degree
numel(cones::CompositeConeGPU{T}) where {T}  = cones.numel

# # -----------------------------------------------------
# # dispatch operators for multiple cones
# # -----------------------------------------------------

function is_symmetric(cones::CompositeConeGPU{T}) where {T}
    #true if all pieces are symmetric.  
    #determined during obj construction
    return cones._is_symmetric
end

function allows_primal_dual_scaling(cones::CompositeConeGPU{T}) where {T}
    all(allows_primal_dual_scaling, cones)
end


# function rectify_equilibration!(
#     cones::CompositeCone{T},
#      δ::AbstractVector{T},
#      e::AbstractVector{T}
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
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    Z = cones.workmat1

    α = cones.α
    @. α = αmin
    
    CUDA.@allowscalar for i in cones.idx_inq
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

    if n_psd > 0
        n_shift = n_linear + n_soc

        svec_to_mat_gpu!(Z,z,rng_cones,n_shift,n_psd)

        # Batched SVD decomposition
        # YC: memory of e can be optimized later
        e = CUDA.CUSOLVER.syevjBatched!('N','U',Z)      #'N' returns eigenvalues only; 'V' returns both eigenvalues and eigenvectors
        αmin = min(αmin,minimum(e))
        @. e = max(e,zero(T))
        CUDA.synchronize()
        β += sum(e)
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
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    rng_cones = cones.rng_cones

    CUDA.@allowscalar begin
        for i in cones.idx_eq
            rng_cone_i = rng_cones[i]
            @views scaled_unit_shift_zero!(z[rng_cone_i],pd)
        end
        for i in cones.idx_inq
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

    if n_psd > 0
        n_shift = n_linear + n_soc
        kernel = @cuda launch=false _kernel_scaled_unit_shift_psd!(z,α,rng_cones,psd_dim,n_shift,n_psd)
        config = launch_configuration(kernel.fun)
        threads = min(n_psd, config.threads)
        blocks = cld(n_psd, threads)

        CUDA.@sync kernel(z,α,rng_cones,psd_dim,n_shift,n_psd; threads, blocks)
    end

    return nothing

end

# unit initialization for asymmetric solves
function unit_initialization!(
    cones::CompositeConeGPU{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim

    αp = cones.αp
    rng_cones = cones.rng_cones

    CUDA.@allowscalar begin
        for i in cones.idx_eq
            rng_cone_i = rng_cones[i]
            @views unit_initialization_zero!(z[rng_cone_i],s[rng_cone_i])
        end
        for i in cones.idx_inq
            rng_cone_i = rng_cones[i]
            @views unit_initialization_nonnegative!(z[rng_cone_i],s[rng_cone_i])
        end
        CUDA.synchronize()
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_unit_initialization_soc!(z,s,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(z,s,rng_cones,n_linear,n_soc; threads, blocks)
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        kernel = @cuda launch=false _kernel_unit_initialization_exp!(z,s,rng_cones,n_shift,n_exp)
        config = launch_configuration(kernel.fun)
        threads = min(n_exp, config.threads)
        blocks = cld(n_exp, threads)

        CUDA.@sync kernel(z,s,rng_cones,n_shift,n_exp; threads, blocks)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        kernel = @cuda launch=false _kernel_unit_initialization_pow!(z,s,αp,rng_cones,n_shift,n_pow)
        config = launch_configuration(kernel.fun)
        threads = min(n_pow, config.threads)
        blocks = cld(n_pow, threads)

        CUDA.@sync kernel(z,s,αp,rng_cones,n_shift,n_pow; threads, blocks)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        unit_initialization_psd_gpu!(z,s,rng_cones,psd_dim,n_shift,n_psd)
    end

    return nothing

end

function set_identity_scaling!(
    cones::CompositeConeGPU{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    w = cones.w
    η = cones.η
    R = cones.R
    Rinv = cones.Rinv
    Hspsd = cones.Hspsd
    psd_dim = cones.psd_dim
    
    CUDA.@allowscalar for i in cones.idx_inq
        @views set_identity_scaling_nonnegative!(w[rng_cones[i]])
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_set_identity_scaling_soc!(w,η,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(w,η,rng_cones,n_linear,n_soc; threads, blocks)
    end

    if n_psd > 0
        kernel = @cuda launch=false _kernel_set_identity_scaling_psd!(R,Rinv,Hspsd,psd_dim,n_psd)
        config = launch_configuration(kernel.fun)
        threads = min(n_psd, config.threads)
        blocks = cld(n_psd, threads)

        CUDA.@sync kernel(R,Rinv,Hspsd,psd_dim,n_psd; threads, blocks)
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
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    αp = cones.αp
    grad = cones.grad
    Hs = cones.Hs
    H_dual = cones.H_dual
    rng_cones = cones.rng_cones
    w = cones.w
    λ = cones.λ
    η = cones.η
    is_scaling_success = true
    
    CUDA.@allowscalar for i in cones.idx_inq
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

    if n_exp > 0
        n_shift = n_linear+n_soc
        kernel = @cuda launch=false _kernel_update_scaling_exp!(s,z,grad,Hs,H_dual,rng_cones,μ,scaling_strategy,n_shift,n_exp)
        config = launch_configuration(kernel.fun)
        threads = min(n_exp, config.threads)
        blocks = cld(n_exp, threads)

        CUDA.@sync kernel(s,z,grad,Hs,H_dual,rng_cones,μ,scaling_strategy,n_shift,n_exp; threads, blocks)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        kernel = @cuda launch=false _kernel_update_scaling_pow!(s,z,grad,Hs,H_dual,αp,rng_cones,μ,scaling_strategy,n_shift,n_exp,n_pow)
        config = launch_configuration(kernel.fun)
        threads = min(n_pow, config.threads)
        blocks = cld(n_pow, threads)

        CUDA.@sync kernel(s,z,grad,Hs,H_dual,αp,rng_cones,μ,scaling_strategy,n_shift,n_exp,n_pow; threads, blocks)
    end

    if n_psd > 0
        L1 = cones.chol1
        L2 = cones.chol2

        n_shift = n_linear+n_soc+n_exp+n_pow

        svec_to_mat_gpu!(L2,z,rng_cones,n_shift,n_psd)
        svec_to_mat_gpu!(L1,s,rng_cones,n_shift,n_psd)

        _, infoz = potrfBatched!(L2, 'L')
        _, infos = potrfBatched!(L1, 'L')

        # YC: This is an issue related to the batched Cholesky factorization in CUSOLVER,
        # which fills in both lower and upper triangular of each submatrix during factorization
        #Set upper triangular parts to 0
        mask_zeros!(L2, 'U')
        mask_zeros!(L1, 'U')

        # bail if the cholesky factorization fails
        if !(all(==(0), infoz) && all(==(0), infos))
            return is_scaling_success = false
        end

        #SVD of L2'*L1,
        tmp = cones.workmat1;
        CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(T), L2, L1, zero(T), tmp)
        U, S, V = CUDA.CUSOLVER.gesvdj!('V', tmp)

        #assemble λ (diagonal), R and Rinv.
        λpsd = cones.λpsd
        Λisqrt = cones.Λisqrt
        copyto!(λpsd, S)
        CUDA.@sync @. Λisqrt = inv.(sqrt.(λpsd))

        #R = L1*(V)*Λisqrt  Λisqrt is a diagonal matrix for each psd cone
        R = cones.R
        Rinv = cones.Rinv
        CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), L1, V, zero(T), R)
        right_mul_batched!(R,Λisqrt,R)

        #Rinv = Λisqrt*(U)'*L2'
        CUDA.CUBLAS.gemm_strided_batched!('T', 'T', one(T), U, L2, zero(T), Rinv)
        left_mul_batched!(Λisqrt,Rinv,Rinv)

        #compute R*R' (upper triangular part only)
        RRt = cones.workmat1;
        fill!(RRt, zero(T))
        CUDA.CUBLAS.gemm_strided_batched!('N', 'T', one(T), R, R, zero(T), RRt)

        #YC: RRt may not be symmetric, not sure how much effect it will be
        Hspsd = cones.Hspsd
        skron_batched!(Hspsd,RRt)
    end

    return is_scaling_success = true
end

# Update Hs block for each cone.
function get_Hs!(
    cones::CompositeConeGPU{T},
    Hsblocks::AbstractVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    Hs = cones.Hs
    Hspsd = cones.Hspsd
    rng_blocks = cones.rng_blocks
    rng_cones = cones.rng_cones
    w = cones.w

    CUDA.@allowscalar begin
        for i in cones.idx_eq
            @views get_Hs_zero!(Hsblocks[rng_blocks[i]])
        end
        for i in cones.idx_inq
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

    if n_exp > 0
        n_shift = n_linear+n_soc
        kernel = @cuda launch=false _kernel_get_Hs_exp!(Hsblocks,Hs,rng_blocks,n_shift,n_exp)
        config = launch_configuration(kernel.fun)
        threads = min(n_exp, config.threads)
        blocks = cld(n_exp, threads)

        CUDA.@sync kernel(Hsblocks,Hs,rng_blocks,n_shift,n_exp; threads, blocks)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        kernel = @cuda launch=false _kernel_get_Hs_pow!(Hsblocks,Hs,rng_blocks,n_shift,n_exp,n_pow)
        config = launch_configuration(kernel.fun)
        threads = min(n_pow, config.threads)
        blocks = cld(n_pow, threads)

        CUDA.@sync kernel(Hsblocks,Hs,rng_blocks,n_shift,n_exp,n_pow; threads, blocks)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        kernel = @cuda launch=false _kernel_get_Hs_psd!(Hsblocks,Hspsd,rng_blocks,n_shift,n_psd)
        config = launch_configuration(kernel.fun)
        threads = min(n_psd, config.threads)
        blocks = cld(n_psd, threads)

        CUDA.@sync kernel(Hsblocks,Hspsd,rng_blocks,n_shift,n_psd; threads, blocks)
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
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    Hs = cones.Hs
    rng_cones = cones.rng_cones
    w = cones.w
    η = cones.η

    CUDA.@allowscalar begin
        for i in cones.idx_eq
            @views mul_Hs_zero!(y[rng_cones[i]])
        end
        for i in cones.idx_inq
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

    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0
        n_shift = n_linear+n_soc
        kernel = @cuda launch=false _kernel_mul_Hs_nonsymmetric!(y,Hs,x,rng_cones,n_shift,n_nonsymmetric)
        config = launch_configuration(kernel.fun)
        threads = min(n_nonsymmetric, config.threads)
        blocks = cld(n_nonsymmetric, threads)

        CUDA.@sync kernel(y,Hs,x,rng_cones,n_shift,n_nonsymmetric; threads, blocks)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow

        CUDA.@allowscalar rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop

        #Transform it into the matrix form 
        @views tmpx = x[rng]
        @views tmpy = y[rng]

        n_tri_dim = triangular_number(psd_dim)
        n_psd_int64 = Int64(n_psd)

        X = reshape(tmpx, (n_tri_dim, n_psd_int64))
        Y = reshape(tmpy, (n_tri_dim, n_psd_int64))

        Hspsd = cones.Hspsd
        CUDA.CUBLAS.gemv_strided_batched!('N',1.0,Hspsd,X,0.0,Y)
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
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    λ = cones.λ

    CUDA.@allowscalar begin
        for i in cones.idx_eq
            @views affine_ds_zero!(ds[rng_cones[i]])
        end
        for i in cones.idx_inq
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

    #update nonsymmetric cones
    if n_exp+n_pow > 0
        n_shift = n_linear + n_soc
        CUDA.@allowscalar begin
            rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_exp+n_pow].stop
        end

        CUDA.@sync @. ds[rng] = s[rng]
    end

    if n_psd > 0
        n_shift = n_linear + n_soc + n_exp + n_pow
        psd_dim = cones.psd_dim
        λpsd = cones.λpsd
        affine_ds_psd_gpu!(ds,λpsd,rng_cones,psd_dim,n_shift,n_psd)
    end

    return nothing
end

function combined_ds_shift!(
    cones::CompositeConeGPU{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    z::AbstractVector{T},
    σμ::T
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    grad = cones.grad
    H_dual = cones.H_dual
    αp = cones.αp
    rng_cones = cones.rng_cones
    w = cones.w
    η = cones.η

    CUDA.synchronize()
    
    CUDA.@allowscalar begin
        for i in cones.idx_eq
            @views combined_ds_shift_zero!(shift[rng_cones[i]])
        end
        for i in cones.idx_inq
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

    if n_exp > 0
        n_shift = n_linear+n_soc
        kernel = @cuda launch=false _kernel_combined_ds_shift_exp!(shift,step_z,step_s,z,grad,H_dual,rng_cones,σμ,n_shift,n_exp)
        config = launch_configuration(kernel.fun)
        threads = min(n_exp, config.threads)
        blocks = cld(n_exp, threads)

        CUDA.@sync kernel(shift,step_z,step_s,z,grad,H_dual,rng_cones,σμ,n_shift,n_exp; threads, blocks)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        kernel = @cuda launch=false _kernel_combined_ds_shift_pow!(shift,step_z,step_s,z,grad,H_dual,αp,rng_cones,σμ,n_shift,n_exp,n_pow)
        config = launch_configuration(kernel.fun)
        threads = min(n_pow, config.threads)
        blocks = cld(n_pow, threads)

        CUDA.@sync kernel(shift,step_z,step_s,z,grad,H_dual,αp,rng_cones,σμ,n_shift,n_exp,n_pow; threads, blocks)
    end
    
    if n_psd > 0 
        n_shift = n_linear+n_soc+n_exp+n_pow
        combined_ds_shift_psd!(cones,shift,step_z,step_s,n_shift,n_psd,σμ)
    end

    return nothing

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
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    w = cones.w
    λ = cones.λ
    η = cones.η

    CUDA.@allowscalar begin
        for i in cones.idx_eq
            @views Δs_from_Δz_offset_zero!(out[rng_cones[i]])
        end
        for i in cones.idx_inq
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

    if n_exp+n_pow > 0
        n_shift = n_linear + n_soc
        CUDA.@allowscalar begin
            rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_exp+n_pow].stop
        end
        @. out[rng] = ds[rng]
        CUDA.synchronize()
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        Δs_from_Δz_offset_psd!(cones, out, ds, work, n_shift, n_psd)
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

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones       = cones.rng_cones
    α               = cones.α
    αp              = cones.αp

    CUDA.@sync @. α = αmax          #Initialize step size

    CUDA.@allowscalar begin
        for i in cones.idx_inq
            len_nn = Cint(length(rng_cones[i]))
            rng_cone_i = rng_cones[i]
            @views dzi = dz[rng_cone_i]
            @views dsi = ds[rng_cone_i]
            @views zi = z[rng_cone_i]
            @views si = s[rng_cone_i]
            @views αi = α[rng_cone_i]
            αc = step_length_nonnegative(dzi,dsi,zi,si,αi,len_nn,αmax)
            αmax = min(αmax,αc)
        end
    end

    if n_soc > 0
        kernel = @cuda launch=false _kernel_step_length_soc(dz,ds,z,s,α,rng_cones,n_linear,n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(dz,ds,z,s,α,rng_cones,n_linear,n_soc; threads, blocks)
        @views αmax = min(αmax,minimum(α[1:n_soc]))
        if αmax < 0
            throw(DomainError("starting point of line search not in SOC"))
        end
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow

        Λisqrt = cones.Λisqrt
        d      = cones.workvec
        Rx     = cones.R
        Rinv   = cones.Rinv
        workΔ  = cones.workmat1
        (workmat1, workmat2, workmat3) = (cones.workmat1, cones.workmat2, cones.workmat3)

        #d = Δz̃ = WΔz
        # We need an extra parameter since the dimension of d is not equal to that of dz
        # αz = step_length_psd_component(workΔ,d,Λisqrt,αmax)
        mul_Wx_psd!(d, dz, Rx, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, true)
        αz = step_length_psd_component_gpu(workΔ,d,Λisqrt,n_psd,αmax)
        
        #d = Δs̃ = W^{-T}Δs
        mul_WTx_psd!(d, ds, Rinv, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, true)
        αs = step_length_psd_component_gpu(workΔ,d,Λisqrt,n_psd,αmax)
        @views αmax = min(αmax,αz,αs)

        if αmax < 0
            throw(DomainError("starting point of line search not in positive semidefinite cones"))
        end
    end

    step = settings.linesearch_backtrack_step
    αmin = settings.min_terminate_step_length
    #if we have any nonsymmetric cones, then back off from full steps slightly
    #so that centrality checks and logarithms don't fail right at the boundaries
    if(n_exp + n_pow > 0)
        αmax = min(αmax,settings.max_step_fraction)
    end
  
    if n_exp > 0
        n_shift = n_linear+n_soc
        kernel = @cuda launch=false _kernel_step_length_exp(dz,ds,z,s,α,rng_cones,αmax,αmin,step,n_shift,n_exp)
        config = launch_configuration(kernel.fun)
        threads = min(n_exp, config.threads)
        blocks = cld(n_exp, threads)

        CUDA.@sync kernel(dz,ds,z,s,α,rng_cones,αmax,αmin,step,n_shift,n_exp; threads, blocks)
        @views αmax = min(αmax,minimum(α[1:n_exp]))
        if αmax < 0
            throw(DomainError("starting point of line search not in expotential cones"))
        end
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        kernel = @cuda launch=false _kernel_step_length_pow(dz,ds,z,s,α,αp,rng_cones,αmax,αmin,step,n_shift,n_pow)
        config = launch_configuration(kernel.fun)
        threads = min(n_pow, config.threads)
        blocks = cld(n_pow, threads)

        CUDA.@sync kernel(dz,ds,z,s,α,αp,rng_cones,αmax,αmin,step,n_shift,n_pow; threads, blocks)
        @views αmax = min(αmax,minimum(α[1:n_pow]))
        if αmax < 0
            throw(DomainError("starting point of line search not in power cones"))
        end
    end

    return (αmax,αmax)
end

# compute the total barrier function at the point (z + α⋅dz, s + α⋅ds)
function compute_barrier(
    cones::CompositeConeGPU{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    n_linear = cones.n_linear
    n_nn = cones.n_nn
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    rng_cones = cones.rng_cones
    αp    = cones.αp
    workmat1 = cones.workmat1
    workvec  = cones.workvec

    barrier = zero(T)
    work = cones.α
    
    if n_nn > 0
        val = compute_barrier_nonnegative(work, z, s, dz, ds, α, rng_cones, cones.idx_inq, n_nn)
        barrier += val
    end

    if n_soc > 0
        val = compute_barrier_soc(work, z, s, dz, ds, α, rng_cones, n_linear, n_soc)
        barrier += val
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        val = compute_barrier_exp(work, z, s, dz, ds, α, rng_cones, n_shift, n_exp)
        barrier += val
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        val = compute_barrier_pow(work, z, s, dz, ds, α, αp, rng_cones, n_shift, n_pow)
        barrier += val
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        val = compute_barrier_psd(work, z, s, dz, ds, α, workmat1, workvec, rng_cones, psd_dim, n_shift, n_psd)
        barrier += val
    end

    return barrier
end

