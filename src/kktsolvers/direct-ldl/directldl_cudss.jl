using CUDA, CUDA.CUSPARSE
using CUDSS

export CUDSSDirectLDLSolver
struct CUDSSDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    KKTgpu::AbstractSparseMatrix{T}
    cudssSolver::CUDSS.CudssSolver{T}
    x::AbstractVector{T}
    b::AbstractVector{T}
    

    function CUDSSDirectLDLSolver{T}(KKT::SparseMatrixCSC{T,Int64},Dsigns,settings) where {T}

        dim = LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        # "S" denotes real symmetric and 'U' denotes the upper triangular
        colptr = CuVector(KKT.colptr)
        rowval = CuVector(KKT.rowval)
        nzval = CuVector(KKT.nzval)
        KKTgpu = CUSPARSE.CuSparseMatrixCSR(colptr, rowval, nzval, (dim, dim))
        cudssSolver = CUDSS.CudssSolver(KKTgpu, "S", 'L')
        x = CuVector(zeros(T, dim))
        b = CuVector(zeros(T, dim))

        cudss("analysis", cudssSolver, x, b)
        cudss("factorization", cudssSolver, x, b)

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
    @views ldlsolver.KKTgpu.nzVal[CuVector(index)] .= CuVector(values)

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
    cudss_set(ldlsolver.cudssSolver.matrix,ldlsolver.KKTgpu)

    # Refactorization
    cudss("factorization", ldlsolver.cudssSolver, ldlsolver.x, ldlsolver.b)

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
    copyto!(bgpu,b)
    
    #solve on GPU
    ldiv!(xgpu,ldlsolver.cudssSolver,bgpu)

    #copy back to CPU
    copyto!(x,xgpu)

end
