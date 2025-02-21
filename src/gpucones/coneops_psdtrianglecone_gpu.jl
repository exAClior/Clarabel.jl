using GenericLinearAlgebra  # extends SVD, eigs etc for BigFloats

# # ----------------------------------------------------
# # Positive Semidefinite Cone
# # ----------------------------------------------------

# degree(K::PSDTriangleCone{T}) where {T} = K.n        #side dimension, M \in \mathcal{S}^{n×n}
# numel(K::PSDTriangleCone{T})  where {T} = K.numel    #number of elements

# function margins is implemented explicitly into the compositecone operation

# place vector into sdp cone
function _kernel_scaled_unit_shift_psd!(
    z::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where{T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        # #adds αI to the vectorized triangle,
        # #at elements [1,3,6....n(n+1)/2]
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views zi = z[rng_cone_i] 
        @inbounds for k = 1:psd_dim
            zi[triangular_index(k)] += α
        end
    end

    return nothing
end

@inline function scaled_unit_shift_psd!(
    z::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_scaled_unit_shift_psd!(z,α,rng_cones,psd_dim,n_shift,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(z,α,rng_cones,psd_dim,n_shift,n_psd; threads, blocks)
end

# unit initialization for asymmetric solves
@inline function unit_initialization_psd_gpu!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
 ) where{T}
 
    CUDA.@allowscalar begin
        rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop
        @views fill!(s[rng],zero(T))
        @views fill!(z[rng],zero(T))
    end
    α = one(T)

    scaled_unit_shift_psd!(z,α,rng_cones,psd_dim,n_shift,n_psd)
    scaled_unit_shift_psd!(s,α,rng_cones,psd_dim,n_shift,n_psd)
 
    return nothing
 end



function _kernel_set_identity_scaling_psd!(
    R::AbstractArray{T,3},
    Rinv::AbstractArray{T,3},
    Hspsd::AbstractArray{T,3},
    psd_dim::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        #Other entries of R, Rinv, Hspsd to 0's
        @inbounds for k in 1:psd_dim
            R[k,k,i] = one(T)
            Rinv[k,k,i] = one(T)
        end
        @inbounds for k in 1:triangular_number(psd_dim)     
            Hspsd[k,k,i] = one(T)
        end
    end

    return nothing
end

function _kernel_get_Hs_psd!(
    Hsblock::AbstractVector{T},
    Hs::AbstractArray{T,3},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        @views Hsi = Hs[:,:,i]
        @views Hsblocki = Hsblock[rng_i]
        
        @inbounds for j in 1:length(Hsblocki)
            Hsblocki[j] = Hsi[j]
        end
    end

    return nothing

end

# # compute the product y = WᵀWx
# function mul_Hs!()
# end

# returns ds = λ ∘ λ for the SDP cone
function _kernel_affine_ds_psd!(
    ds::AbstractVector{T},
    λpsd::AbstractMatrix{T},
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        #We have Λ = Diagonal(K.λ), so
        #λ ∘ λ should map to Λ.^2
        shift_idx = rng_cones[n_shift+i].start - 1
        #same as X = Λ*Λ
        @inbounds for k = 1:psd_dim
            ds[shift_idx+triangular_index(k)] = λpsd[k,i]^2
        end
    end

    return nothing
end

@inline function affine_ds_psd_gpu!(
    ds::AbstractVector{T},
    λpsd::AbstractMatrix{T},
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    CUDA.@allowscalar begin
        rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop
    end
    @views fill!(ds[rng],zero(T))

    kernel = @cuda launch=false _kernel_affine_ds_psd!(ds,λpsd,rng_cones,psd_dim,n_shift,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(ds,λpsd,rng_cones,psd_dim,n_shift,n_psd; threads, blocks)
end

@inline function combined_ds_shift_psd!(
    cones::CompositeConeGPU{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    n_shift::Cint,
    n_psd::Cint,
    σμ::T
) where {T}

    #shift vector used as workspace for a few steps 
    tmp = shift    
    R = cones.R
    Rinv = cones.Rinv
    rng_cones = cones.rng_cones
    workmat1 = cones.workmat1
    workmat2 = cones.workmat2
    workmat3 = cones.workmat3
    psd_dim = cones.psd_dim
    
    CUDA.@allowscalar begin
        rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop
    end

     #Δz <- WΔz
    CUDA.@sync @. tmp[rng] = step_z[rng]        
    mul_Wx_psd!(step_z, tmp, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false)     

    #Δs <- W⁻TΔs
    CUDA.@sync @. tmp[rng] = step_s[rng]            
    mul_WTx_psd!(step_s, tmp, Rinv, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false)   

    #shift = W⁻¹Δs ∘ WΔz - σμe
    #X  .= (Y*Z + Z*Y)/2 
    # NB: Y and Z are both symmetric
    svec_to_mat_gpu!(workmat1,step_z,rng_cones,n_shift,n_psd)
    svec_to_mat_gpu!(workmat2,step_s,rng_cones,n_shift,n_psd)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), workmat1, workmat2, zero(T), workmat3)
    symmetric_part_gpu!(workmat3)

    mat_to_svec_gpu!(shift,workmat3,rng_cones,n_shift,n_psd)       
    
    scaled_unit_shift_psd!(shift,-σμ,rng_cones,psd_dim,n_shift,n_psd)                     

    return nothing

end


function _kernel_op_λ!(
    X::AbstractArray{T,3},
    Z::AbstractArray{T,3},
    λpsd::AbstractMatrix{T},
    psd_dim::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        @views Xi = X[:,:,i]
        @views Zi = Z[:,:,i]
        @views λi = λpsd[:,i]
        for k = 1:psd_dim
            for j = 1:psd_dim
                Xi[k,j] = 2*Zi[k,j]/(λi[k] + λi[j])
            end
        end
    end

    return nothing
end

@inline function op_λ!(
    X::AbstractArray{T,3},
    Z::AbstractArray{T,3},
    λpsd::AbstractMatrix{T},
    psd_dim::Cint,
    n_psd::Cint
) where {T}

    kernel = @cuda launch=false _kernel_op_λ!(X, Z, λpsd, psd_dim, n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(X, Z, λpsd, psd_dim, n_psd; threads, blocks)
end

@inline function Δs_from_Δz_offset_psd!(
    cones::CompositeConeGPU{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    n_shift::Cint,
    n_psd::Cint
) where {T}

    R = cones.R
    λpsd = cones.λpsd
    rng_cones = cones.rng_cones
    workmat1 = cones.workmat1
    workmat2 = cones.workmat2
    workmat3 = cones.workmat3
    psd_dim = cones.psd_dim

    #tmp = λ \ ds 
    svec_to_mat_gpu!(workmat2, ds, rng_cones, n_shift, n_psd)
    op_λ!(workmat1, workmat2, λpsd, psd_dim, n_psd)
    mat_to_svec_gpu!(work, workmat1, rng_cones, n_shift, n_psd) 

    #out = Wᵀ(λ \ ds) = Wᵀ(work) 
    mul_WTx_psd!(out, work, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false)   

end

function _kernel_logdet!(
    barrier::AbstractVector{T},
    fact::AbstractArray{T,3},
    psd_dim::Cint,
    n_psd::Cint
) where {T}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        val = zero(T)
        @inbounds for k = 1:psd_dim
            val += logsafe(fact[k,k,i])
        end
        barrier[i] = val + val
    end

    return nothing
end

@inline function _logdet_barrier_psd(
    barrier::AbstractVector{T},
    x::AbstractVector{T},
    dx::AbstractVector{T},
    alpha::T,
    workmat1::AbstractArray{T,3},
    workvec::AbstractVector{T},
    rng::UnitRange{Cint},
    psd_dim::Cint,
    n_psd::Cint
) where {T}

    Q = workmat1
    q = workvec

    CUDA.@sync @. q = x[rng] + alpha*dx[rng]
    svec_to_mat_no_shift_gpu!(Q, q, n_psd)
    _, info = potrfBatched!(Q, 'L')


    if all(==(0), info)
        kernel = @cuda launch=false _kernel_logdet!(barrier, Q, psd_dim, n_psd)
        config = launch_configuration(kernel.fun)
        threads = min(n_psd, config.threads)
        blocks = cld(n_psd, threads)
    
        CUDA.@sync kernel(barrier, Q, psd_dim, n_psd; threads, blocks)

        return sum(@view barrier[1:n_psd])
    else 
        return typemax(T)
    end

end

@inline function compute_barrier_psd(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    workmat1::AbstractArray{T,3},
    workvec::AbstractVector{T},
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    CUDA.@allowscalar begin
        rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop
    end

    barrier_d = _logdet_barrier_psd(barrier, z, dz, α, workmat1, workvec, rng, psd_dim, n_psd)
    barrier_p = _logdet_barrier_psd(barrier, s, ds, α, workmat1, workvec, rng, psd_dim, n_psd)
    return (- barrier_d - barrier_p)

end

# ---------------------------------------------
# operations supported by symmetric cones only 
# ---------------------------------------------

# implements y = Wx for the PSD cone
@inline function mul_Wx_psd!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    Rx::AbstractArray{T,3},
    rng_cones::AbstractVector,
    workmat1::AbstractArray{T,3},
    workmat2::AbstractArray{T,3},
    workmat3::AbstractArray{T,3},
    n_shift::Cint,
    n_psd::Cint,
    step_search::Bool
) where {T}

    (X,Y,tmp) = (workmat1,workmat2,workmat3)

    svec_to_mat_gpu!(X,x,rng_cones,n_shift,n_psd)

    #Y .= (R'*X*R)                #W*x
    # mul!(tmp,Rx',X)
    # mul!(Y,tmp,Rx)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(T), Rx, X, zero(T), tmp)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), tmp, Rx, zero(T), Y)

    (step_search) ? mat_to_svec_no_shift_gpu!(y,Y,n_psd) : mat_to_svec_gpu!(y,Y,rng_cones,n_shift,n_psd)

    return nothing
end

# implements y = WTx for the PSD cone
@inline function mul_WTx_psd!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    Rx::AbstractArray{T,3},
    rng_cones::AbstractVector,
    workmat1::AbstractArray{T,3},
    workmat2::AbstractArray{T,3},
    workmat3::AbstractArray{T,3},
    n_shift::Cint,
    n_psd::Cint,
    step_search::Bool
) where {T}

    (X,Y,tmp) = (workmat1,workmat2,workmat3)

    svec_to_mat_gpu!(X,x,rng_cones,n_shift,n_psd)

    #Y .= (R*X*R')                #W^T*x 
    # mul!(tmp,X,Rx')
    # mul!(Y,Rx,tmp)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', one(T), X, Rx, zero(T), tmp)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), Rx, tmp, zero(T), Y)

    (step_search) ? mat_to_svec_no_shift_gpu!(y,Y,n_psd) : mat_to_svec_gpu!(y,Y,rng_cones,n_shift,n_psd)

    return nothing
end

# # implements y = αW^{-1}x + βy for the psd cone
# function mul_Winv!(
#     K::PSDTriangleCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#     mul_Wx_inner(
#         is_transpose,
#         y,x,
#         α,
#         β,
#         K.data.Rinv,
#         K.data.workmat1,
#         K.data.workmat2,
#         K.data.workmat3)
    
# end

# # implements x = λ \ z for the SDP cone
# function λ_inv_circ_op!(
#     K::PSDTriangleCone{T},
#     x::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     (X,Z) = (K.data.workmat1,K.data.workmat2)
#     map((M,v)->svec_to_mat!(M,v),(X,Z),(x,z))

#     λ = K.data.λ
#     for i = 1:K.n
#         for j = 1:K.n
#             X[i,j] = 2*Z[i,j]/(λ[i] + λ[j])
#         end
#     end
#     mat_to_svec!(x,X)

#     return nothing
# end

# # ---------------------------------------------
# # Jordan algebra operations for symmetric cones 
# # ---------------------------------------------

# # implements x = y ∘ z for the SDP cone
# function circ_op!(
#     K::PSDTriangleCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     (Y,Z) = (K.data.workmat1,K.data.workmat2)
#     map((M,v)->svec_to_mat!(M,v),(Y,Z),(y,z))

#     X = K.data.workmat3;

#     #X  .= (Y*Z + Z*Y)/2 
#     # NB: Y and Z are both symmetric
#     if T <: LinearAlgebra.BlasFloat
#         LinearAlgebra.BLAS.syr2k!('U', 'N', T(0.5), Y, Z, zero(T), X)
#     else 
#         X .= (Y*Z + Z*Y)/2
#     end
#     mat_to_svec!(x,Symmetric(X))

#     return nothing
# end

# # implements x = y \ z for the SDP cone
# function inv_circ_op!(
#     K::PSDTriangleCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     # X should be the solution to (YX + XY)/2 = Z

#     #  For general arguments this requires solution to a symmetric
#     # Sylvester equation.  Throwing an error here since I do not think
#     # the inverse of the ∘ operator is ever required for general arguments,
#     # and solving this equation is best avoided.

#     error("This function not implemented and should never be reached.")

#     return nothing
# end

#-----------------------------------------
# internal operations for SDP cones 
# ----------------------------------------

function step_length_psd_component_gpu(
    workΔ::AbstractArray{T,3},
    d::AbstractVector{T},
    Λisqrt::AbstractMatrix{T},
    n_psd::Cint,
    αmax::T
) where {T}
    
    # NB: this could be made faster since we only need to populate the upper triangle 
    svec_to_mat_no_shift_gpu!(workΔ,d,n_psd)
    lrscale_gpu!(workΔ,Λisqrt)
    # symmetric_part_gpu!(workΔ)

    # batched eigenvalue decomposition
    e = CUDA.CUSOLVER.syevjBatched!('N','U',workΔ)

    γ = minimum(e)
    if γ < 0
        return min(inv(-γ),αmax)
    else
        return αmax
    end

end

#make a matrix view from a vectorized input
function _kernel_svec_to_mat!(
    Z::AbstractArray{T,3}, 
    z::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        @views Zi = Z[:,:,i]
        @views zi = z[rng_i]
        svec_to_mat!(Zi,zi)
    end

    return nothing
end

@inline function svec_to_mat_gpu!(
    Z::AbstractArray{T,3}, 
    z::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_svec_to_mat!(Z,z,rng_blocks,n_shift,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(Z,z,rng_blocks,n_shift,n_psd; threads, blocks)
end

#No shift version of svec_to_mat
function _kernel_svec_to_mat_no_shift!(
    Z::AbstractArray{T,3}, 
    z::AbstractVector{T},
    n_psd::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        @views Zi = Z[:,:,i]
        dim = size(Zi,1)
        rng_i = ((i-1)*triangular_number(dim) + 1):(i*triangular_number(dim))
        @views zi = z[rng_i]
        svec_to_mat!(Zi,zi)
    end

    return nothing
end

@inline function svec_to_mat_no_shift_gpu!(
    Z::AbstractArray{T,3}, 
    z::AbstractVector{T},
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_svec_to_mat_no_shift!(Z,z,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(Z,z,n_psd; threads, blocks)
end

#make a matrix view from a vectorized input
function _kernel_mat_to_svec!(
    z::AbstractVector{T},
    Z::AbstractArray{T,3}, 
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        @views Zi = Z[:,:,i]
        @views zi = z[rng_i]
        mat_to_svec!(zi,Zi)
    end

    return nothing
end

@inline function mat_to_svec_gpu!(
    z::AbstractVector{T},
    Z::AbstractArray{T,3}, 
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_mat_to_svec!(z,Z,rng_blocks,n_shift,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(z,Z,rng_blocks,n_shift,n_psd; threads, blocks)
end

#No shift version of mat_to_svec
function _kernel_mat_to_svec_no_shift!(
    z::AbstractVector{T},
    Z::AbstractArray{T,3}, 
    n_psd::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_psd
        @views Zi = Z[:,:,i]
        dim = size(Zi,1)
        rng_i = ((i-1)*triangular_number(dim) + 1):(i*triangular_number(dim))
        @views zi = z[rng_i]
        mat_to_svec!(zi,Zi)
    end

    return nothing
end

@inline function mat_to_svec_no_shift_gpu!(
    z::AbstractVector{T},
    Z::AbstractArray{T,3}, 
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_mat_to_svec_no_shift!(z,Z,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(z,Z,n_psd; threads, blocks)
end

# produce the upper triangular part of the Symmetric Kronecker product of
# a symmtric matrix A with itself, i.e. triu(A ⊗_s A) with full fill-in
function skron_full!(
    out::AbstractMatrix{T},
    A::AbstractMatrix{T},
) where {T}

    sqrt2  = sqrt(2)
    n      = size(A, 1)

    col = 1
    @inbounds for l in 1:n
        @inbounds for k in 1:l
            row = 1
            kl_eq = k == l

            @inbounds for j in 1:n
                Ajl = A[j, l]
                Ajk = A[j, k]

                @inbounds for i in 1:j
                    (row > col) && break
                    ij_eq = i == j

                    if (ij_eq, kl_eq) == (false, false)
                        out[row, col] = A[i, k] * Ajl + A[i, l] * Ajk
                    elseif (ij_eq, kl_eq) == (true, false) 
                        out[row, col] = sqrt2 * Ajl * Ajk
                    elseif (ij_eq, kl_eq) == (false, true)  
                        out[row, col] = sqrt2 * A[i, l] * Ajk
                    else (ij_eq,kl_eq) == (true, true)
                        out[row, col] = Ajl * Ajl
                    end 

                    #Also fill-in the lower triangular part
                    out[col, row] = out[row, col]

                    row += 1
                end # i
            end # j
            col += 1
        end # k
    end # l
end

function _kernel_skron!(
    out::AbstractArray{T,3}, 
    A::AbstractArray{T,3},
    n::Clong
) where {T}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        @views outi = out[:,:,i]
        @views Ai = A[:,:,i]

        skron_full!(outi,Ai)
    end

    return nothing
end

@inline function skron_batched!(
    out::AbstractArray{T,3}, 
    A::AbstractArray{T,3}
) where {T}
    n = size(out,3)

    kernel = @cuda launch=false _kernel_skron!(out,A,n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(out,A,n; threads, blocks)
end

#right multiplication for A[:,:,i] with the diagonal matrix of B[:,i]
function _kernel_right_mul!(
    A::AbstractArray{T,3}, 
    B::AbstractArray{T,2},
    C::AbstractArray{T,3},
    n2::Cint,
    n::Cint
) where {T}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        (k,j) = divrem(i-1, n2)
        j += 1
        k += 1
        val = B[j,k]
        @inbounds for l in axes(A,1)
            C[l,j,k] = val*A[l,j,k]
        end
    end

    return nothing
end

@inline function right_mul_batched!(
    A::AbstractArray{T,3}, 
    B::AbstractArray{T,2},
    C::AbstractArray{T,3}
) where {T}
    n2 = Cint(size(A,2))
    n = n2*Cint(size(A,3))

    kernel = @cuda launch=false _kernel_right_mul!(A,B,C,n2,n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A,B,C,n2,n; threads, blocks)
end

#left multiplication for B[:,:,i] with the diagonal matrix of A[:,i]
function _kernel_left_mul!(
    A::AbstractArray{T,2}, 
    B::AbstractArray{T,3},
    C::AbstractArray{T,3},
    n2::Cint,
    n::Cint
) where {T}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        (k,j) = divrem(i-1, n2)
        j += 1
        k += 1
        val = A[j,k]
        @inbounds for l in axes(A,1)
            C[j,l,k] = val*B[j,l,k]
        end
    end

    return nothing
end

@inline function left_mul_batched!(
    A::AbstractArray{T,2}, 
    B::AbstractArray{T,3},
    C::AbstractArray{T,3}
) where {T}
    n2 = Cint(size(B,2))
    n = n2*Cint(size(B,3))

    kernel = @cuda launch=false _kernel_left_mul!(A,B,C,n2,n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A,B,C,n2,n; threads, blocks)
end