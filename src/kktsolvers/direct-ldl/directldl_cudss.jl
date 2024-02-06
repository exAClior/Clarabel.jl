using CUDA, CUDA.CUSPARSE
using CUDSS

struct CUDSSDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    KKTgpu::AbstractSparseMatrix{T}
    cudssSolver::CUDSS.CudssSolver{T}
    x::CudssMatrix{T}
    b::CudssMatrix{T}
    

    function CUDSSDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        dim = LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        # "S" denotes real symmetric and 'U' denotes the upper triangular
        KKTgpu = CuSparseMatrixCSR(KKT)
        cudssSolver = CUDSS.CudssSolver(KKTgpu, "S", 'U')
        x = CudssMatrix(T, n)
        b = CudssMatrix(T, n)

        return new(KKTgpu,cudssSolver,x,b)
    end

end

DirectLDLSolversDict[:cudss] = CUDSSDirectLDLSolver
required_matrix_shape(::Type{CUDSSDirectLDLSolver}) = :triu

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::CUDSSDirectLDLSolver{T},
    index::AbstractVector{Int},
    values::Vector{T}
) where{T}

    #Update values that are stored within KKTgpu
    @views ldlsolver.KKTgpu.nzVal[index] .= values

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::CUDSSDirectLDLSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}

    @views ldlsolver.KKTgpu.nzVal[index] .*= scale

end


#refactor the linear system
function refactor!(ldlsolver::CUDSSDirectLDLSolver{T}, K::SparseMatrixCSC) where{T}

    # Update the KKT matrix in the cudss solver
    cudss_set(ldlsolver.cudssSolver,ldlsolver.KKTgpu)

    # YC: should be corrected later on 
    return true
    # return all(isfinite, cudss_get(ldlsolver.cudssSolver.data,"diag"))

end


#solve the linear system
function solve!(
    ldlsolver::CUDSSDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    #solve in place 
    xgpu = ldlsolver.x
    bgpu = ldlsolver.b
    @. bgpu = b
    
    #solve on GPU
    ldiv!(xgpu,ldlsolver.cudssSolver,bgpu)

    #copy back to CPU
    @. x = xgpu

end
