using Krylov
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using LinearOperators

struct MINRESIndirectSolver{T} <: AbstractIndirectMINRESSolver{T}

    solver#::MinresQlpSolver{T,T,AbstractVector{T}}
    KKT::AbstractSparseMatrix{T}
    KKTcpu::SparseMatrixCSC{T,Int}
    A::SparseMatrixCSC{T,Int}

    M::AbstractPreconditioner{T} #YC: Right now, we use the diagonal preconditioner, 
                                    #    but we need to find a better option 
    b::AbstractVector{T}
    work::AbstractVector{T}                 #YC: work space for preconditioner

    #YC: additional parameter
    preconditioner::Int             #0 means disabled; 1 is the partial preconditioner and 2 is the norm preconditioner.
    cpu::Bool
    atol::T                         #YC: using iterative refinement atol and rtol values at present
    rtol::T

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function MINRESIndirectSolver{T}(KKT0::Symmetric{T, SparseMatrixCSC{T,Int}}, settings,m,n) where {T}
        
        dim = m + n
        cpu = (settings.device == :cpu) ? true : false;
        Vectype = (cpu) ? Vector : CuVector;
        M = DiagonalPreconditioner{T}(Vectype(ones(T,dim)),m)

        #YC: We want to use CuSparseMatrixCSR instead of a Symmetric matrix to ensure faster computation
        KKTcpu = SparseMatrixCSC(KKT0)      #YC: temporary use
        KKT = (cpu) ? KKTcpu : CuSparseMatrixCSR(KKTcpu);     
        A = KKTcpu[n+1:dim,1:n]

        b = Vectype(ones(T,dim))
        work = ones(T,dim)

        solver = MinresQlpSolver(dim,dim,Vectype{T})

        preconditioner = settings.preconditioner

        return new(solver,KKT,KKTcpu,A,M,b,work, preconditioner,cpu,settings.iterative_refinement_abstol,settings.iterative_refinement_reltol)
    end

end

IndirectMINRESSolversDict[:minres] = MINRESIndirectSolver
required_matrix_shape(::Type{MINRESIndirectSolver}) = :triu


#refactor the linear system
function refactor!(minressolver::MINRESIndirectSolver{T}, KKT::SparseMatrixCSC, diagval::AbstractVector{T}) where{T}

        #YC: simple copy of data with redundant copy for P, A, A' parts to GPU-KKT
        fill_lower_triangular_symmetric!(KKT, minressolver.KKTcpu)

        update_preconditioner(minressolver, diagval)  #YC: update the preconditioner

        if (minressolver.cpu)    #cpu
            return is_success = true
        else
            copyto!(minressolver.KKT.nzVal, minressolver.KKTcpu.nzval)  #gpu
        end
    
        # @assert issymmetric(minressolver.KKTcpu)   #YC: need to be removed later    

    return is_success = true # required to compile for some reason

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

    # #Preconditioner is pending
    if (minressolver.preconditioner > 0)
        minres_qlp!(minressolver.solver,KKT, minressolver.b; M = minressolver.M, ldiv = true, atol= minressolver.atol, rtol= minressolver.atol) # ,verbose= Int64(floor(length(b)/10)),history=true)  # verbose=n,history=true
    else
        minres_qlp!(minressolver.solver,KKT, minressolver.b; atol= minressolver.atol, rtol= minressolver.rtol)
    end

    println("Iter: ", minressolver.solver.stats.niter)

    # println("Iter num", minressolver.solver.stats.niter)
    copyto!(x, minressolver.solver.x)

    # todo: note that the second output of minres is the stats, perhaps might be useful to the user

end


###############################################
# YC: temporary function for GPU implementation 
###############################################
function fill_lower_triangular_symmetric!(A::SparseMatrixCSC{T,Ti}, B::SparseMatrixCSC{T,Ti}) where {T,Ti}
    # @assert LinearAlgebra.istriu(A)

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