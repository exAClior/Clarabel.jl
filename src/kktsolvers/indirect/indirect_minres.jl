struct MINRESIndirectSolver{T} <: AbstractIndirectSolver{T}

    solver#::MinresQlpSolver{T,T,AbstractVector{T}}
    KKT::AbstractSparseMatrix{T}
    KKTcpu::SparseMatrixCSC{T,Int}
    A::SparseMatrixCSC{T,Int}

    M::AbstractPreconditioner{T} #YC: Right now, we use the diagonal preconditioner, 
                                    #    but we need to find a better option 
    b::AbstractVector{T}
    work::AbstractVector{T}                 #YC: work space for preconditioner

    #YC: additional parameter
    m::Int                          # number of constraints
    preconditioner::Int             #0 means disabled; 1 is the partial preconditioner and 2 is the norm preconditioner.
    cpu::Bool
    atol::T                         #YC: using iterative refinement atol and rtol values at present
    rtol::T

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function MINRESIndirectSolver{T}(KKT0::Symmetric{T, SparseMatrixCSC{T,Int}}, settings,m,n) where {T}
        
        dim = m + n
        cpu = (settings.device == :cpu) ? true : false;
        Vectype = (cpu) ? Vector : CuVector;
        M = DiagonalPreconditioner{T}(Vectype(ones(T,dim)))

        #YC: We want to use CuSparseMatrixCSR instead of a Symmetric matrix to ensure faster computation
        KKTcpu = SparseMatrixCSC(KKT0)      #YC: temporary use
        KKT = (cpu) ? KKTcpu : CuSparseMatrixCSR(KKTcpu);     
        A = KKTcpu[n+1:dim,1:n]

        b = Vectype(ones(T,dim))
        work = ones(T,dim)

        solver = MinresQlpSolver(dim,dim,Vectype{T})

        preconditioner = settings.preconditioner
        atol = settings.iterative_refinement_abstol
        rtol = settings.iterative_refinement_reltol

        return new(solver,KKT,KKTcpu,A,M,b,work,
                m,preconditioner,cpu,
                atol,rtol)
    end

end

IndirectSolversDict[:minres] = MINRESIndirectSolver
required_matrix_shape(::Type{MINRESIndirectSolver}) = :triu

# Diagonal Preconditioning
function update_preconditioner(
    solver::MINRESIndirectSolver{T},
    diagval::AbstractVector{T}
) where {T}
    KKT = solver.KKTcpu
    M = solver.M 
    preconditioner = solver.preconditioner
    A = solver.A
    work = M.work
    dim = length(work)
    m = solver.m

    #Diagonal partial preconditioner
    if (preconditioner == 1)
        n = dim - m
        workm = @view work[(dim-m+1):dim]
        diagvalm = @view diagval[(dim-m+1):dim]
        workn = @view work[1:n]
        diagvaln = @view diagval[1:n]
        
        @. workm = -one(T)/diagvalm         # preconditioner for the constraint part, now assuming only linear inequality constraints

        #compute diagonals of A'*H^{-1}*A
        @. workn = zero(T)

        @inbounds Threads.@threads for i= 1:n
            tmp = zero(T)
            #traverse the column j in A
            @inbounds for j= A.colptr[i]:(A.colptr[i+1]-1)  
                tmp += A.nzval[j]^2*workm[A.rowval[j]]
            end
            workn[i] = tmp
        end

        @. workn = one(T)/(workn + diagvaln)      # preconditioner for the variable part
    end

    # Diagonal norm preconditioner, but we can make it more efficient
    if (preconditioner == 2)
        @. work = one(T)
        @inbounds Threads.@threads for i = 1:dim
            KKTcol = view(KKT.nzval,KKT.colptr[i]:(KKT.colptr[i+1]-1))
            work[i] = one(T)/max(work[i],norm(KKTcol,Inf))
        end
    end

    @assert all(work .> zero(T))       #preconditioner need to be p.s.d.

    copyto!(M.diagval,work)
end




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