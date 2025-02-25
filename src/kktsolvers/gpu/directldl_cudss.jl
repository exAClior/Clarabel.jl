using CUDA, CUDA.CUSPARSE
using CUDSS

export CUDSSDirectLDLSolver
struct CUDSSDirectLDLSolver{T} <: AbstractGPUSolver{T}

    KKT::AbstractSparseMatrix{T}
    cudssSolver::CUDSS.CudssSolver{T}
    x::AbstractVector{T}
    b::AbstractVector{T}
    

    function CUDSSDirectLDLSolver{T}(KKT::AbstractSparseMatrix{T},x,b) where {T}

        LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        # "S" denotes real symmetric and 'U' denotes the upper triangular

        cudssSolver = CUDSS.CudssSolver(KKT, "S", 'F')

        cudss("analysis", cudssSolver, x, b)
        cudss("factorization", cudssSolver, x, b)

        return new(KKT,cudssSolver,x,b)
    end

end

GPUSolversDict[:cudss] = CUDSSDirectLDLSolver
required_matrix_shape(::Type{CUDSSDirectLDLSolver}) = :full

#refactor the linear system
function refactor!(ldlsolver::CUDSSDirectLDLSolver{T}) where{T}

    # Update the KKT matrix in the cudss solver
    cudss_set(ldlsolver.cudssSolver.matrix,ldlsolver.KKT)

    # Refactorization
    cudss("factorization", ldlsolver.cudssSolver, ldlsolver.x, ldlsolver.b)

    # YC: should be corrected later on 
    return true
    # return all(isfinite, cudss_get(ldlsolver.cudssSolver.data,"diag"))

end


#solve the linear system
function solve!(
    ldlsolver::CUDSSDirectLDLSolver{T},
    x::AbstractVector{T},
    b::AbstractVector{T}
) where{T}
    
    #solve on GPU
    ldiv!(x,ldlsolver.cudssSolver,b)

end
