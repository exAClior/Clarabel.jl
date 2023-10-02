struct MINRESIndirectSolver{T} <: AbstractIndirectSolver{T}

    solver#::MinresQlpSolver{T,T,AbstractVector{T}}
    KKT::AbstractSparseMatrix{T}
    KKTcpu::SparseMatrixCSC{T,Int}

    b::AbstractVector{T}
    work::AbstractVector{T}                 #YC: work space for preconditioner

    #YC: additional parameter
    cpu::Bool
    atol::T                         #YC: using iterative refinement atol and rtol values at present
    rtol::T

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function MINRESIndirectSolver{T}(KKT0::SparseMatrixCSC{T,Int}, settings) where {T}
        
        cpu = (settings.device == :cpu) ? true : false;
        Vectype = (cpu) ? Vector : CuVector;

        #YC: KKTcpu shares the same memory with KKT0
        KKTcpu = KKT0
        KKT = (cpu) ? KKTcpu : CuSparseMatrixCSR(KKTcpu);     #We use CUDA if gpu is selected
        
        dim = KKTcpu.n
        b = Vectype(ones(T,dim))
        work = ones(T,dim)

        solver = MinresQlpSolver(dim,dim,Vectype{T})

        atol = settings.iterative_refinement_abstol
        rtol = settings.iterative_refinement_reltol

        return new(solver,KKT,KKTcpu,b,work,
                cpu,atol,rtol)
    end

end

IndirectSolversDict[:minres] = MINRESIndirectSolver
required_matrix_shape(::Type{MINRESIndirectSolver}) = :full

#refactor the linear system
function refactor!(minressolver::MINRESIndirectSolver{T}) where{T}

        if (minressolver.cpu)    #cpu
            return is_success = true
        else
            copyto!(minressolver.KKT.nzVal, minressolver.KKTcpu.nzval)  #gpu
        end

    return is_success = true # required to compile for some reason

end

#solve the linear system
function solve!(
    minressolver::MINRESIndirectSolver{T},
    preconditioner::Union{AbstractPreconditioner{T},UniformScaling{Bool}},
    x::AbstractVector{T},
    b::AbstractVector{T}
) where{T}

    KKT = minressolver.KKT

    #data copy to gpu
    copyto!(minressolver.b,b)

    #Preconditioner
    minres_qlp!(minressolver.solver,KKT, minressolver.b; M = preconditioner, ldiv = true, atol= minressolver.atol, rtol= minressolver.rtol)#,verbose= Int64(floor(length(b)/10)),history=true)  # verbose=n,history=true

    # println("minres error: ", minressolver.solver.stats.niter)

    # println("Iter num", minressolver.solver.stats.niter)
    copyto!(x, minressolver.solver.x)

    # todo: note that the second output of minres is the stats, perhaps might be useful to the user

end
