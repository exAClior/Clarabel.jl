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
    p::Int                          # number of augmented constraints
    preconditioner::Int             #0 means disabled; 1 is the partial preconditioner and 2 is the norm preconditioner.
    cpu::Bool
    atol::T                         #YC: using iterative refinement atol and rtol values at present
    rtol::T

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function MINRESIndirectSolver{T}(KKT0::SparseMatrixCSC{T,Int}, settings,m,n,p) where {T}
        
        dim = m + n + p
        cpu = (settings.device == :cpu) ? true : false;
        Vectype = (cpu) ? Vector : CuVector;
        M = DiagonalPreconditioner{T}(Vectype(ones(T,dim)))

        #YC: KKTcpu shares the same memory with KKT0
        KKTcpu = KKT0
        KKT = (cpu) ? KKTcpu : CuSparseMatrixCSR(KKTcpu);     #We use CUDA if gpu is selected
        A = KKTcpu[n+1:dim,1:n]

        b = Vectype(ones(T,dim))
        work = ones(T,dim)

        solver = MinresQlpSolver(dim,dim,Vectype{T})

        preconditioner = settings.preconditioner
        atol = settings.iterative_refinement_abstol
        rtol = settings.iterative_refinement_reltol

        return new(solver,KKT,KKTcpu,A,M,b,work,
                m,p,preconditioner,cpu,
                atol,rtol)
    end

end

IndirectSolversDict[:minres] = MINRESIndirectSolver
required_matrix_shape(::Type{MINRESIndirectSolver}) = :full

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
    p = solver.p

    #Diagonal partial preconditioner
    if (preconditioner == 1)
        n = dim - m - p
        workn = @view work[1:n]
        diagvaln = @view diagval[1:n]
        workm = @view work[(n+1):(n+m)]
        diagvalm = @view diagval[(n+1):(n+m)]
        
        @. workm = -one(T)/diagvalm         # preconditioner for the constraint part, now assuming only linear inequality constraints
        #inverse the absolute value for augmented diagonals
        if (p > zero(T))
            workp = @view work[n+m+1:end]
            diagvalp = @view diagval[n+m+1:end]
            @. workp =  one(T)/abs(diagvalp) 
        end

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
