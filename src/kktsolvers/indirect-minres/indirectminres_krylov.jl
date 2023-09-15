using Krylov
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using LinearOperators

struct MINRESIndirectSolver{T} <: AbstractIndirectMINRESSolver{T}

    solver#::MinresQlpSolver{T,T,AbstractVector{T}}
    KKT::AbstractSparseMatrix{T}
    KKTcpu::SparseMatrixCSC{T,Int}

    Pinv::AbstractSparseMatrix{T}       #YC: Right now, we use the diagonal preconditioner, 
                                    #    but we need to find a better option 
    b::AbstractVector{T}
    work::AbstractVector{T}                 #YC: work space for preconditioner

    #YC: additional parameter
    preconditioner::Bool
    cpu::Bool
    atol::T                         #YC: using iterative refinement atol and rtol values at present
    rtol::T

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function MINRESIndirectSolver{T}(KKT0::Symmetric{T, SparseMatrixCSC{T,Int}}, Dsigns, settings) where {T}
        
        dim = LinearAlgebra.checksquare(KKT0) # just to check if square

        cpu = (settings.device == :cpu) ? true : false;
        Vectype = (cpu) ? Vector : CuVector;
        Pinv = (cpu) ? spdiagm(ones(T,dim)) : CuSparseMatrixCSR(spdiagm(ones(T,dim))); 

        #YC: We want to use CuSparseMatrixCSR instead of a Symmetric matrix to ensure faster computation
        KKTcpu = SparseMatrixCSC(KKT0)      #YC: temporary use
        KKT = (cpu) ? KKTcpu : CuSparseMatrixCSR(KKTcpu);     

        b = Vectype(ones(T,dim))
        work = ones(T,dim)

        solver = MinresQlpSolver(dim,dim,Vectype{T})

        preconditioner = settings.preconditioner

        return new(solver,KKT,KKTcpu,Pinv,b,work, preconditioner,cpu,settings.iterative_refinement_abstol,settings.iterative_refinement_reltol)
    end

end

IndirectMINRESSolversDict[:minres] = MINRESIndirectSolver
required_matrix_shape(::Type{MINRESIndirectSolver}) = :triu


#refactor the linear system
function refactor!(minressolver::MINRESIndirectSolver{T}, KKT::SparseMatrixCSC) where{T}

        #YC: simple copy of data with redundant copy for P, A, A' parts to GPU-KKT
        fill_lower_triangular_symmetric!(KKT, minressolver.KKTcpu)

        if (minressolver.cpu)    #cpu
            return is_success = true
        else
            copyto!(minressolver.KKT.nzVal, minressolver.KKTcpu.nzval)  #gpu
        end
    
        @assert issymmetric(minressolver.KKTcpu)   #YC: need to be removed later    

    return is_success = true # required to compile for some reason

end

function update_preconditioner(
    minressolver:: MINRESIndirectSolver{T},
    KKT::SparseMatrixCSC{T,Ti},
    index::Vector{Ti},
    m::Int      #number of constraints
) where {T, Ti}
    work = minressolver.work
    Pinv = minressolver.Pinv
    dim = length(work)
    n = dim - m

    @inbounds for i=(dim-m+1):dim
        work[i] = one(T) / ( - KKT[i,i])    # preconditioner for the constraint part, now assuming only linear inequality constraints
    end

    #compute diagonals of A'*H^{-1}*A
    workn = @view work[1:n]
    @. workn = zero(T)
    @inbounds for i= 1:m
        diaginv = one(T)/(-KKT.nzval[index[n+i]])

        At = @view KKT[1:n,n+i]
        @. workn += At*At*diaginv
    end

    # At = @view KKT[1:n,(n+1):dim]

    @inbounds for i = 1:n
        work[i] = one(T)/(KKT[i,i] + work[i])     # preconditioner for the variable part
    end

    @assert all(work .> zero(T))       #preconditioner need to be p.s.d.

    if (minressolver.cpu)
        copyto!(Pinv.nzval,work)
    else
        copyto!(Pinv.nzVal,work)    #copy to GPU workspace
    end
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
    if (minressolver.preconditioner)
        minres_qlp!(minressolver.solver,KKT, minressolver.b; M = minressolver.Pinv, atol= minressolver.atol, rtol= minressolver.atol)# ,verbose=length(b),history=true)  # verbose=n,history=true
    else
        minres_qlp!(minressolver.solver,KKT, minressolver.b; atol= minressolver.atol, rtol= minressolver.rtol)
    end

    # println("Iter num", minressolver.solver.stats.niter)
    copyto!(x, minressolver.solver.x)

    # todo: note that the second output of minres is the stats, perhaps might be useful to the user

end


###############################################
# YC: temporary function for GPU implementation 
###############################################
function fill_lower_triangular_symmetric!(A::SparseMatrixCSC{T,Ti}, B::SparseMatrixCSC{T,Ti}) where {T,Ti}
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