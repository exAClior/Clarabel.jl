using CUDA, CUDA.CUSPARSE
using CUDSS

export CUDSSDirectLDLSolverMixed
struct CUDSSDirectLDLSolverMixed{T} <: AbstractGPUSolver{T}

    KKTgpu::AbstractSparseMatrix{T}
    cudssSolver::CUDSS.CudssSolver{Float32}

    KKTFloat32::AbstractSparseMatrix{Float32}
    xFloat32::AbstractVector{Float32}
    bFloat32::AbstractVector{Float32}
    

    function CUDSSDirectLDLSolverMixed{T}(KKT::AbstractSparseMatrix{T},x,b) where {T}

        dim = LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        # "S" denotes real symmetric and 'U' denotes the upper triangular

        KKTgpu = KKT

        val = CuVector{Float32}(KKTgpu.nzVal)
        KKTFloat32 = CuSparseMatrixCSR(KKTgpu.rowPtr,KKTgpu.colVal,val,size(KKTgpu))
        cudssSolver = CUDSS.CudssSolver(KKTFloat32, "S", 'F')

        xFloat32 = CUDA.zeros(Float32,dim)
        bFloat32 = CUDA.zeros(Float32,dim)

        cudss("analysis", cudssSolver, xFloat32, bFloat32)
        cudss("factorization", cudssSolver, xFloat32, bFloat32)


        return new(KKTgpu,cudssSolver,KKTFloat32,xFloat32,bFloat32)
    end

end

GPUSolversDict[:cudssmixed] = CUDSSDirectLDLSolverMixed
required_matrix_shape(::Type{CUDSSDirectLDLSolverMixed}) = :full

#refactor the linear system
function refactor!(ldlsolver::CUDSSDirectLDLSolverMixed{T}) where{T}

    #YC: Copy data from a Float64 matrix to Float32 matrix
    copyto!(ldlsolver.KKTFloat32.nzVal,ldlsolver.KKTgpu.nzVal)

    # Update the KKT matrix in the cudss solver
    cudss_set(ldlsolver.cudssSolver.matrix,ldlsolver.KKTFloat32)

    # Refactorization
    cudss("factorization", ldlsolver.cudssSolver, ldlsolver.xFloat32, ldlsolver.bFloat32)

    # YC: should be corrected later on 
    return true
    # return all(isfinite, cudss_get(ldlsolver.cudssSolver.data,"diag"))

end

#solve the linear system
function solve!(
    ldlsolver::CUDSSDirectLDLSolverMixed{T},
    x::AbstractVector{T},
    b::AbstractVector{T}
) where{T}

    xFloat32 = ldlsolver.xFloat32
    bFloat32 = ldlsolver.bFloat32

    #convert b to Float32
    copyto!(bFloat32, b)

    #solve on GPU
    ldiv!(xFloat32, ldlsolver.cudssSolver, bFloat32)

    #convert to Float64, copy to x 
    copyto!(x, xFloat32)

end
