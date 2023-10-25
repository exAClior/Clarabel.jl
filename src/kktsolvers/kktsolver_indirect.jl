# -------------------------------------
# KKTSolver using indirect solvers
# -------------------------------------

##############################################################
# YC: Some functions are repeated as in the direct solver, which are better to be removed
##############################################################
mutable struct IndirectKKTSolver{T} <: AbstractKKTSolver{T}

    # problem dimensions
    m::Int; n::Int; p::Int

    # Left and right hand sides for solves
    x::AbstractVector{T}
    b::AbstractVector{T}

    # internal workspace for IR scheme
    # and static offsetting of KKT
    work1::AbstractVector{T}
    work2::AbstractVector{T}

    #KKT mapping from problem data to KKT
    map::IndirectDataMap 

    #the expected signs of D in KKT = LDL^T
    Dsigns::Vector{Int}

    # a vector for storing the Hs blocks
    # on the in the KKT matrix block diagonal
    Hsblocks::Vector{Vector{T}}
    cones::CompositeCone{T}

    #unpermuted KKT matrix
    KKT::SparseMatrixCSC{T,Int}

    #settings just points back to the main solver settings.
    #Required since there is no separate LDL settings container
    settings::Settings{T}

    #the direct linear LDL solver
    indirectsolver::AbstractIndirectSolver{T}

    #the diagonal regularizer currently applied
    diagonal_regularizer::T


    #YC: additional parameters
    A::SparseMatrixCSC{T,Int}
    M::Union{AbstractPreconditioner{T},UniformScaling{Bool}} #YC: Right now, we use the diagonal preconditioner, 
                                    #    but we need to find a better option 


    function IndirectKKTSolver{T}(P,A,cones,m,n,settings) where {T}

        # get a constructor for the LDL solver we should use,
        # and also the matrix shape it requires
        (kktshape, indirectsolverT, preconditionerT) = _get_indirectsolver_config(settings)
        Vectype = ((settings.device == :cpu)) ? Vector : CuVector;

        #construct a KKT matrix of the right shape
        KKT, map = _assemble_full_kkt_matrix(P,A,cones,kktshape)

        #Need this many extra variables for sparse cones
        p = pdim(map.sparse_maps)

        #LHS/RHS/work for iterative refinement
        x    = Vector{T}(undef,n+m+p)
        b    = Vector{T}(undef,n+m+p)
        work_e  = Vector{T}(undef,n+m+p)
        work_dx = Vector{T}(undef,n+m+p)

        #the expected signs of D in LDL
        Dsigns = Vector{Int}(undef,n+m+p)
        _fill_Dsigns!(Dsigns,m,n,map)

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        dim = m + n + p
        A = KKT[n+1:dim,1:n]

        Hsblocks = _allocate_full_kkt_Hsblocks(T, cones)

        M = preconditionerT{T,Vectype}(m,n,KKT)

        diagonal_regularizer = zero(T)

        #the indirect linear solver engine
        indirectsolver = indirectsolverT{T}(KKT,A,settings)

        return new(m,n,p,x,b,
                   work_e,work_dx,map,Dsigns,Hsblocks,cones,
                   KKT,settings,indirectsolver,
                   diagonal_regularizer,
                   A,M
                   )
    end

end

IndirectKKTSolver(args...) = IndirectKKTSolver{DefaultFloat}(args...)

function _get_indirectsolver_type(s::Symbol)
    try
        return IndirectSolversDict[s]
    catch
        throw(error("Unsupported indirect linear solver :", s))
    end
end

function _get_preconditioner_type(s::Int)
    try
        return PreconditionersDict[s]
    catch
        throw(error("Unsupported indirect linear solver :", s))
    end
end

function _get_indirectsolver_config(settings::Settings)

    #which LDL solver should I use?
    indirectsolverT = _get_indirectsolver_type(settings.indirect_solve_method)

    #does it want a :triu or :tril KKT matrix?
    kktshape = required_matrix_shape(indirectsolverT)

    #Get the type of preconditioner
    preconditionerT = _get_preconditioner_type(settings.preconditioner)

    (kktshape,indirectsolverT,preconditionerT)
end 


#update entries in the kktsolver object using the
#given index into its CSC representation
function _update_values!(
    indirectsolver::AbstractIndirectSolver{T},
    KKT::SparseMatrixCSC{T,Ti},
    index::AbstractVector{Ti},
    values::AbstractVector{T}
) where{T,Ti}

    #YC: should tailored when using GPU
    #Update values in the KKT matrix K
    @. KKT.nzval[index] = values

end


#scale entries in the kktsolver object using the
#given index into its CSC representation
function _scale_values!(
    indirectsolver::AbstractIndirectSolver{T},
    KKT::SparseMatrixCSC{T,Ti},
    index::Vector{Ti},
    scale::T
) where{T,Ti}

    #YC: should tailored when using GPU
    #Update values in the KKT matrix K
    @. KKT.nzval[index] *= scale

end


function kktsolver_update!(
    kktsolver:: IndirectKKTSolver{T},
    cones::CompositeCone{T}
) where {T}

    # the internal  indirectsolver is type unstable, so multiple
    # calls to the  indirectsolvers will be very slow if called
    # directly.   Grab it here and then call an inner function
    # so that the  indirectsolver has concrete type
    indirectsolver = kktsolver.indirectsolver
    return _kktsolver_update_inner!(kktsolver,indirectsolver,cones)
end


function _kktsolver_update_inner!(
    kktsolver:: IndirectKKTSolver{T},
    indirectsolver::AbstractIndirectSolver{T},
    cones::CompositeCone{T}
) where {T}

    #real implementation is here, and now  indirectsolver
    #will be compiled to something concrete.

    map       = kktsolver.map
    KKT       = kktsolver.KKT

    #Set the elements the W^tW blocks in the KKT matrix.
    get_Hs!(cones,kktsolver.Hsblocks,false)

    for (index, values) in zip(map.Hsblocks,kktsolver.Hsblocks)
        #change signs to get -W^TW
        @. values *= -one(T)
        _update_values!(indirectsolver,KKT,index,values)
    end

    sparse_map_iter = Iterators.Stateful(map.sparse_maps)

    updateFcn = (index,values) -> _update_values!(indirectsolver,KKT,index,values)
    scaleFcn  = (index,scale)  -> _scale_values!(indirectsolver,KKT,index,scale)

    for cone in cones
        #add sparse expansions columns for sparse cones 
        if @conedispatch is_sparse_expandable(cone)
            thismap = popfirst!(sparse_map_iter)
            _csc_update_sparsecone_full(cone,thismap,updateFcn,scaleFcn)
        end 
    end

    return _kktsolver_regularize_and_refactor!(kktsolver,  indirectsolver)

end

function _kktsolver_regularize_and_refactor!(
    kktsolver::IndirectKKTSolver{T},
    indirectsolver::AbstractIndirectSolver{T}
) where{T}

    settings      = kktsolver.settings
    map           = kktsolver.map
    KKT           = kktsolver.KKT
    Dsigns        = kktsolver.Dsigns
    diag_kkt      = kktsolver.work1
    diag_shifted  = kktsolver.work2

    if(settings.static_regularization_enable)

        # hold a copy of the true KKT diagonal
        @views diag_kkt .= KKT.nzval[map.diag_full]
        ϵ = _compute_regularizer(diag_kkt, settings)

        # compute an offset version, accounting for signs
        diag_shifted .= diag_kkt
        @inbounds for i in eachindex(Dsigns)
            if(Dsigns[i] == 1) diag_shifted[i] += ϵ
            else               diag_shifted[i] -= ϵ
            end
        end
        # overwrite the diagonal of KKT and within the  indirectsolver
        _update_values!(indirectsolver,KKT,map.diag_full,diag_shifted)

        # remember the value we used.  Not needed,
        # but possibly useful for debugging
        kktsolver.diagonal_regularizer = ϵ

    end

    # YC: we copy KKT information here to the indirect solver,
    # but update_preconditioner in the indirect methods
    
    if !(kktsolver.M === I)
        update_preconditioner!(kktsolver, kktsolver.M, diag_shifted)  #YC: update the preconditioner
    end

    is_success = refactor!(indirectsolver)

    if(settings.static_regularization_enable)

        # put our internal copy of the KKT matrix back the way
        # it was. Not necessary to fix the  indirectsolver copy because
        # this is only needed for our post-factorization IR scheme

        _update_values_KKT!(KKT,map.diag_full,diag_kkt)

    end

    return is_success
end


# function _compute_regularizer(
#     diag_kkt::AbstractVector{T},
#     settings::Settings{T}
# ) where {T}

#     maxdiag  = norm(diag_kkt,Inf);

#     # Compute a new regularizer
#     regularizer =  settings.static_regularization_constant +
#                    settings.static_regularization_proportional * maxdiag

#     return regularizer

# end


function kktsolver_setrhs!(
    kktsolver::IndirectKKTSolver{T},
    rhsx::AbstractVector{T},
    rhsz::AbstractVector{T}
) where {T}

    b = kktsolver.b
    (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)

    b[1:n]             .= rhsx
    b[(n+1):(n+m)]     .= rhsz
    b[(n+m+1):(n+m+p)] .= 0

    return nothing
end


function kktsolver_getlhs!(
    kktsolver::IndirectKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    x = kktsolver.x
    (m,n) = (kktsolver.m,kktsolver.n)

    isnothing(lhsx) || (@views lhsx .= x[1:n])
    isnothing(lhsz) || (@views lhsz .= x[(n+1):(n+m)])

    return nothing
end


function kktsolver_solve!(
    kktsolver::IndirectKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    (x,b) = (kktsolver.x,kktsolver.b)
    solve!(kktsolver.indirectsolver,kktsolver.M,x,b)

    is_success = begin
        if(kktsolver.settings.iterative_refinement_enable)
            #IR reports success based on finite normed residual
            is_success = _iterative_refinement(kktsolver,kktsolver.indirectsolver)
        else
             # otherwise must directly verify finite values
            is_success = all(isfinite,x)
        end
    end

    if is_success
       kktsolver_getlhs!(kktsolver,lhsx,lhsz)
    end

    return is_success
end

# YC: need an efficient refinement as the indirect solver doesn't factorize
#     a system but repeat the multiplication iteratively

#   Moreover, warm-start for the iterative refinement is also an issue

function  _iterative_refinement(
    kktsolver::IndirectKKTSolver{T},
    indirectsolver::AbstractIndirectSolver{T}
) where{T}

    (x,b)   = (kktsolver.x,kktsolver.b)
    (e,dx)  = (kktsolver.work1, kktsolver.work2)
    settings = kktsolver.settings

    #iterative refinement params
    IR_reltol    = settings.iterative_refinement_reltol
    IR_abstol    = settings.iterative_refinement_abstol
    IR_maxiter   = settings.iterative_refinement_max_iter
    IR_stopratio = settings.iterative_refinement_stop_ratio

    KKT = kktsolver.KKT
    normb  = norm(b,Inf)

    #compute the initial error
    norme = _get_refine_error!(e,b,KKT,x)
    isfinite(norme) || return is_success = false

    # println("error is: ", norme)

    for i = 1:IR_maxiter

        if(norme <= IR_abstol + IR_reltol*normb)
            # within tolerance, or failed.  Exit
            break
        end
        lastnorme = norme

        #make a refinement and continue
        solve!(indirectsolver,kktsolver.M,dx,e)

        #prospective solution is x + dx.   Use dx space to
        #hold it for a check before applying to x
        @. dx += x
        norme = _get_refine_error!(e,b,KKT,dx)
        isfinite(norme) || return is_success = false

        improved_ratio = lastnorme/norme
        if(improved_ratio <  IR_stopratio)
            #insufficient improvement.  Exit
            if (improved_ratio > one(T))
                (x,dx) = (dx,x)   #pointer swap
            end
            break
        end
        (x,dx) = (dx,x)           #pointer swap
    end

    # make sure kktsolver fields now point to the right place
    # following possible swaps.   This is not necessary in the
    # Rust implementation since implementation there is via borrow
    (kktsolver.x,kktsolver.work2) = (x,dx)
 
    #NB: "success" means only that we had a finite valued result
    return is_success = true
end


# # computes e = b - Kξ, overwriting the first argument
# # and returning its norm

function _get_refine_error!(
    e::AbstractVector{T},
    b::AbstractVector{T},
    KKT::SparseMatrixCSC{T, Ti},
    ξ::AbstractVector{T}) where {T,Ti}

    @. e = b
    mul!(e,KKT,ξ,-1.,1.)   # e = b - Kξ

    return norm(e,Inf)

end
