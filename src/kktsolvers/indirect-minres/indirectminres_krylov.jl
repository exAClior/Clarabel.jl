using Krylov
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using LinearOperators

struct MINRESIndirectSolver{T} <: AbstractIndirectMINRESSolver{T}

    solver#::MinresQlpSolver{T,T,AbstractVector{T}}
    KKT::AbstractSparseMatrix{T}
    KKTcpu::SparseMatrixCSC{T,Int64}

    preconditioner::AbstractVector{T}       #YC: Right now, we use the diagonal preconditioner, 
                                    #    but we need to find a better option 
    b::AbstractVector{T}
    work::AbstractVector{T}                 #YC: work space for preconditioner

    #YC: warm_start doesn't work at present
    x_constant::AbstractVector{T}       #warm start for the constant solve  
    x_affine::AbstractVector{T}         #warm start for the affine step
    x_combined::AbstractVector{T}       #warm start for the combined step

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function MINRESIndirectSolver{T}(KKT0::Symmetric{T, SparseMatrixCSC{T, Int64}}, Dsigns, settings) where {T}
        
        dim = LinearAlgebra.checksquare(KKT0) # just to check if square

        preconditioner = CuVector(ones(T,dim))

        #YC: We want to use CuSparseMatrixCSR instead of a Symmetric matrix to ensure faster computation
        KKTcpu = SparseMatrixCSC(KKT0)      #YC: temporary use
        KKT = CuSparseMatrixCSR(KKTcpu)     

        b = CuVector(ones(T,dim))
        work = zeros(T,dim)

        solver = MinresQlpSolver(dim,dim,CuVector{T})

        x_constant = CuVector(ones(T,dim))
        x_affine   = CuVector(ones(T,dim))
        x_combined = CuVector(ones(T,dim))

        return new(solver,KKT,KKTcpu,preconditioner,b,work,x_constant,x_affine,x_combined)
    end

end

IndirectMINRESSolversDict[:minres] = MINRESIndirectSolver
required_matrix_shape(::Type{MINRESIndirectSolver}) = :triu


#refactor the linear system
function refactor!(minressolver::MINRESIndirectSolver{T}, KKT::SparseMatrixCSC) where{T}

        #YC: simple copy of data with redundant copy for P, A, A' parts to GPU-KKT
        fill_lower_triangular_symmetric!(KKT, minressolver.KKTcpu)

        copyto!(minressolver.KKT.nzVal, minressolver.KKTcpu.nzval)  
    
        @assert issymmetric(minressolver.KKT)   #YC: need to be removed later    

    return is_success = true # required to compile for some reason

end

function update_preconditioner(
    minressolver:: MINRESIndirectSolver{T},
    KKT::Symmetric{T},
) where {T}
    work = minressolver.work
    n, m = size(KKT)

    @inbounds for i=1:m
        work[i] = one(T) / max(abs(KKT[i,i]),1e-8)    # Jacobi preconditioner
    end

    #copy to GPU workspace
    copyto!(minressolver.preconditioner,work)
end


#solve the linear system
function solve!(
    minressolver::MINRESIndirectSolver{T},
    x::AbstractVector{T},
    b::AbstractVector{T}
) where{T}

    KKT = minressolver.KKT

    #data copy to gpu
    copyto!(minressolver.b,b)

    #solve in place 
    n, m = size(KKT)

    Pinv = Diagonal(minressolver.preconditioner)

    # #Preconditioner is pending
    minres_qlp!(minressolver.solver,KKT, minressolver.b; M = Pinv, Artol=1e-12,atol=1e-12,rtol=1e-13)    # verbose=n,history=true  # verbose=n,history=true
    copyto!(x, minressolver.solver.x)

    # todo: note that the second output of minres is the stats, perhaps might be useful to the user

end


###############################################
# YC: temporary function for GPU implementation 
###############################################
function fill_lower_triangular_symmetric!(A::SparseMatrixCSC{T,Int64}, B::SparseMatrixCSC{T,Int64}) where {T}
    @assert LinearAlgebra.istriu(A)

    n = LinearAlgebra.checksquare(B)
    
    for j in 1:n
        colptr = B.colptr[j]
        colptr_next = B.colptr[j + 1]

        for k in colptr:colptr_next-1
            rowidx = B.rowval[k]

            if rowidx < j
                B[rowidx, j] = A[rowidx, j]
            else
                B[rowidx, j] = A[j, rowidx]
            end
        end
    end
end