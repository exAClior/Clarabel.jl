using CUDA
using CUDA.CUBLAS
using LinearAlgebra

import CUDA.CUBLAS: unsafe_strided_batch, handle
import CUDA.CUBLAS: cublasStatus_t, cublasHandle_t, cublasFillMode_t
import CUDA.CUSOLVER: cusolverDnHandle_t, cusolverStatus_t
import CUDA: unsafe_free!
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo
using Libdl

####################################################################
# Two different types of 3-dimensional arrays are supported:
#   1)CuArray{Float64,3}
#   2)Vector{<:StridedCuMatrix{$elty}}
#   The second structure has more flexibility to support SDP cones of different dimensionalities,
#   see my example for Choleksy factorization
#   Wrappers for different factorizations can be found in
#   1) https://github.com/JuliaGPU/CUDA.jl/blob/7ff012f21ecaf9364a348289a136deebe299e8d9/lib/cusolver/dense.jl
#   2) https://github.com/JuliaGPU/CuArrays.jl/blob/284142de673572fc90578e15c8dce04e5589a17b/src/blas/wrappers.jl
####################################################################

# batch number
N = 100

# Create a vector of matrices (StridedMatrix)
matrices1 = [
    CuArray{Float64,2}(ones(3, 3)) for i = 1:N    # 3x3 matrix
]

matrices2 = [
    CuArray{Float64,2}(i*ones(3, 3)) for i = 1:N    # 3x3 matrix
]

matrices3 = [
    CuArray{Float64,2}(ones(3, 3)) for i = 1:N    # 3x3 matrix
]

########################################################
# Batched general matrix-matrix multiplication 
# # C = beta*C + alpha * A * B
# # transA set to 'N' (no transpose) or 'T' (transpose)
# function gemm_batched!(transA::Char,
#     transB::Char,
#     alpha::Number,
#     A::Vector{<:StridedCuMatrix{$elty}},
#     B::Vector{<:StridedCuMatrix{$elty}},
#     beta::Number,
#     C::Vector{<:StridedCuMatrix{$elty}})
########################################################
CUDA.CUBLAS.gemm_batched!('N', 'N', 1.0, matrices1, matrices2, 0.0, matrices3)


#############################################
# Batched SVD
# function gesvdj!(jobz::Char,
#     econ::Int,
#     A::CuMatrix{$elty};
#     tol::$relty=eps($relty),
#     max_sweeps::Int=100)
#############################################
matrices4 = CuArray{Float64,3}(zeros(3, 3, N))    # Nx3x3 matrix
for i in 1:N
    matrices4[:, :, i] .= CUDA.CuArray([  5.0 0.0 3.0;
                            0.0 2.0 0.0;
                            3.0 0.0 4.0])  # Use broadcasting to assign each 3x3 matrix
end
U, S, V = CUDA.CUSOLVER.gesvdj!('V', matrices4)



#############################################
# Batched LU (in-place decomposition)
#############################################
# getrfBatched - performs LU factorizations
function cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize)
    # find your path via filter(contains("cublas"), Libdl.dllist())
    ccall((:cublasDgetrfBatched, CUDA.libcublas), cublasStatus_t,
                   (cublasHandle_t, Cint, CuPtr{Ptr{Cdouble}}, Cint, CuPtr{Cint},
                    CuPtr{Cint}, Cint),
                   handle, n, A, lda, P, info, batchSize)
end

function cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    ccall((:cublasSgetriBatched, CUDA.libcublas), cublasStatus_t,
                   (cublasHandle_t, Cint, CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint},
                    CuPtr{Ptr{Cfloat}}, Cint, CuPtr{Cint}, Cint),
                   handle, n, A, lda, P, C, ldc, info, batchSize)
end

for (fname, elty) in
    ((:cublasDgetrfBatched,:Float64),
     (:cublasSgetrfBatched,:Float32),
    #  (:cublasZgetrfBatched,:ComplexF64),
    #  (:cublasCgetrfBatched,:ComplexF32)
     )

    @eval begin
        # cublasStatus_t cublasDgetrfBatched(
        #   cublasHandle_t handle, int n, double **A,
        #   int lda, int *PivotArray, int *infoArray,
        #   int batchSize)
        function getrf_batched!(n, ptrs::CuVector{CuPtr{$elty}}, lda, pivot::Bool)
            batchSize = length(ptrs)
            info = CuArray{Cint}(undef, batchSize)
            if pivot
                pivotArray = CuArray{Cint}(undef, (n, batchSize))
                $fname(handle(), n, ptrs, lda, pivotArray, info, batchSize)
            else
                $fname(handle(), n, ptrs, lda, CU_NULL, info, batchSize)
                pivotArray = CUDA.zeros(Cint, (n, batchSize))
            end
            unsafe_free!(ptrs)

            return pivotArray, info
        end
    end
end

# CUDA has no strided batched getrf, but we can at least avoid constructing costly views
function getrf_strided_batched!(A::CuArray{<:Any, 3}, pivot::Bool)
    m,n = size(A,1), size(A,2)
    if m != n
        throw(DimensionMismatch("All matrices must be square!"))
    end
    lda = max(1,stride(A,2))

    Aptrs = unsafe_strided_batch(A)

    return getrf_batched!(n, Aptrs, lda, pivot)..., A
end

matrices5 = CuArray{Float64,3}(zeros(3, 3, N))    # Nx3x3 matrix
for i in 1:N
    matrices5[:, :, i] .= CUDA.CuArray([  5.0 0.0 3.0;
                            0.0 2.0 0.0;
                            3.0 0.0 4.0])  # Use broadcasting to assign each 3x3 matrix
end
matrices6 = deepcopy(matrices5)
(t1,t2,t3) = getrf_strided_batched!(matrices5,false)
# (t4,t5,t6) = CUDA.CUBLAS.getrf_strided_batched!(matrices6,false)


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
        function potrf_strided_batched!(A::CuArray{<:Any, 3},uplo::Char)

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

matrices7 = [
    CUDA.CuArray([  5.0 0.0 3.0;
                            0.0 2.0 0.0;
                            3.0 0.0 4.0])  for i = 1:N    # 3x3 matrix
]
matrices7[100] = CUDA.CuArray([  5.0 0.0;
                                    0.0 2.0]) 
# It shows                          
CUDA.CUSOLVER.potrfBatched!('U', matrices7)

matrices8 = CuArray{Float64,3}(zeros(3, 3, N))    # Nx3x3 matrix
for i in 1:N
    matrices8[:, :, i] .= CUDA.CuArray([  5.0 0.0 3.0;
                            0.0 2.0 0.0;
                            3.0 0.0 4.0])  # Use broadcasting to assign each 3x3 matrix
end
potrf_strided_batched!(matrices8,'U')

