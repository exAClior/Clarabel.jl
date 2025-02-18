import CUDA.CUBLAS: unsafe_strided_batch, handle
import CUDA.CUBLAS: cublasStatus_t, cublasHandle_t, cublasFillMode_t
import CUDA.CUSOLVER: cusolverDnHandle_t, cusolverStatus_t
import CUDA: unsafe_free!
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo
# using Libdl

#############################################
# Batched Cholesky (in-place decomposition)
#############################################
# potrfBatched - performs Cholesky factorizations
#Float64
function cusolverDnDpotrfBatched(handle, uplo, n, A, lda, info, batchSize)
    ccall((:cusolverDnDpotrfBatched, CUDA.libcusolver), cusolverStatus_t, 
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                   CuPtr{Cint}, Cint),
                   handle, uplo, n, A, lda, info, batchSize)
end

#Float32
function cusolverDnSpotrfBatched(handle, uplo, n, A, lda, info, batchSize)
    ccall((:cusolverDnSpotrfBatched, CUDA.libcusolver), cusolverStatus_t, 
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                   CuPtr{Cint}, Cint),
                   handle, uplo, n, A, lda, info, batchSize)
end
for (fname, elty) in ((:cusolverDnSpotrfBatched, :Float32),
    (:cusolverDnDpotrfBatched, :Float64)
    )
    @eval begin
        function potrfBatched!(A::CuArray{$elty, 3},uplo::Char)

            # Set up information for the solver arguments
            chkuplo(uplo)
            n = LinearAlgebra.checksquare(A[:,:,1])
            lda = max(1, stride(A[:,:,1], 2))
            batchSize = size(A,3)

            Aptrs = unsafe_strided_batch(A)

            dh = CUDA.CUSOLVER.dense_handle()
            resize!(dh.info, batchSize)

            # Run the solver
            $fname(dh, uplo, n, Aptrs, lda, dh.info, batchSize)

            # Copy the solver info and delete the device memory
            info = CUDA.@allowscalar collect(dh.info)

            # Double check the solver's exit status
            for i = 1:batchSize
                chkargsok(CUDA.CUSOLVER.BlasInt(info[i]))
            end

            # info[i] > 0 means the leading minor of order info[i] is not positive definite
            # LinearAlgebra.LAPACK does not throw Exception here
            # to simplify calls to isposdef! and factorize
            return A, info
        end
    end
end

#########################################################
# Masked zeros
#########################################################
function _kernel_mask_zeros!(
    A::AbstractArray{T, 3},
    uplo::Char,
    n::Clong
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        @views Ai = A[:,:,i]
        tril!(Ai)
    end

    return nothing
end

@inline function mask_zeros!(
    A::AbstractArray{T, 3},
    uplo::Char
) where {T}
    n = size(A,3)

    kernel = @cuda launch=false _kernel_mask_zeros!(A, uplo, n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A, uplo, n; threads, blocks)
end

#########################################################
# lrscale
#########################################################
function _kernel_lrscale!(
    A::AbstractArray{T, 3},
    d::AbstractMatrix{T},
    n::Clong
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        @views Ai = A[:,:,i]
        @views di = d[:,i]
        lrscale!(di,Ai,di)
    end

    return nothing
end

@inline function lrscale_gpu!(
    A::AbstractArray{T,3},
    d::AbstractMatrix{T}
) where {T}
    n = size(A,3)

    kernel = @cuda launch=false _kernel_lrscale!(A, d, n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A, d, n; threads, blocks)
    
end

#########################################################
# symmetric_part
#########################################################
function _kernel_symmetric_part!(
    A::AbstractArray{T, 3},
    n::Clong
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        @views Ai = A[:,:,i]
        symmetric_part!(Ai)
    end

    return nothing
end

@inline function symmetric_part_gpu!(
    A::AbstractArray{T,3}
) where {T}
    n = size(A,3)

    kernel = @cuda launch=false _kernel_symmetric_part!(A, n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A, n; threads, blocks)
    
end