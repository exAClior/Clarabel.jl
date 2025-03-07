# -------------------------------------
# KKTSolver using cudss direct solvers
# -------------------------------------

const CuVectorView{T} = SubArray{T, 1, AbstractVector{T}, Tuple{AbstractVector{Int}}, false}
##############################################################
# YC: Some functions are repeated as in the direct solver, which are better to be removed
##############################################################
mutable struct GPULDLKKTSolver{T} <: AbstractKKTSolver{T}

    # problem dimensions
    m::Int; n::Int

    # Left and right hand sides for solves
    x::AbstractVector{T}
    b::AbstractVector{T}

    # internal workspace for IR scheme
    # and static offsetting of KKT
    work1::AbstractVector{T}
    work2::AbstractVector{T}

    #KKT mapping from problem data to KKT
    map::GPUDataMap 

    #the expected signs of D in KKT = LDL^T
    Dsigns::AbstractVector{Cint}

    # a vector for storing the Hs blocks
    # on the in the KKT matrix block diagonal
    Hsblocks::AbstractVector{T}

    #unpermuted KKT matrix
    KKT::AbstractCuSparseMatrix{T}

    #settings just points back to the main solver settings.
    #Required since there is no separate LDL settings container
    settings::Settings{T}

    #the direct linear LDL solver
    GPUsolver::AbstractDirectLDLSolver{T}

    #the diagonal regularizer currently applied
    diagonal_regularizer::T

    function GPULDLKKTSolver{T}(
        P::AbstractCuSparseMatrix{T},
        A::AbstractCuSparseMatrix{T},
        At::AbstractCuSparseMatrix{T},
        cones::CompositeConeGPU{T},
        m::Int64,
        n::Int64,
        settings::Settings{T}
    ) where {T}

        # get a constructor for the LDL solver we should use,
        # and also the matrix shape it requires
        (kktshape, GPUsolverT) = get_ldlsolver_config(settings)

        #construct a KKT matrix of the right shape
        KKT, map = _assemble_full_kkt_matrix(P,A,At,cones)

        #YC: disabled sparse expansion and preprocess a large second-order cone into multiple small cones

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        dim = m + n

        #LHS/RHS/work for iterative refinement
        x    = CuVector{T}(undef,dim)
        b    = CuVector{T}(undef,dim)
        work1 = CuVector{T}(undef,dim)
        work2 = CuVector{T}(undef,dim)

        #the expected signs of D in LDL
        Dsigns = CUDA.ones(Cint,dim)
        @views fill!(Dsigns[n+1:n+m], -one(Cint))

        Hsblocks = map.Hsblocks
        diagonal_regularizer = zero(T)

        #the indirect linear solver engine
        GPUsolver = GPUsolverT{T}(KKT,x,b)

        return new(m,n,x,b,
                   work1,work2,map,
                   Dsigns,
                   Hsblocks,
                   KKT,settings,GPUsolver,
                   diagonal_regularizer
                   )
    end

end

GPULDLKKTSolver(args...) = GPULDLKKTSolver{DefaultFloat}(args...)

#update entries in the kktsolver object using the
#given index into its CSC representation
function _update_values!(
    GPUsolver::AbstractDirectLDLSolver{T},
    KKT::CuSparseMatrix{T},
    index::AbstractVector{Ti},
    values::AbstractVector{T}
) where{T,Ti}

    #Update values in the KKT matrix K
    @. KKT.nzVal[index] = values

end

#updates KKT matrix values
function _update_diag_values_KKT!(
    KKT::AbstractCuSparseMatrix{T},
    index::AbstractVector{Ti},
    values::AbstractVector{T}
) where{T,Ti}

    #Update values in the KKT matrix K
    @views copyto!(KKT.nzVal[index], values)
    
end

function kktsolver_update!(
    kktsolver:: GPULDLKKTSolver{T},
    cones::CompositeConeGPU{T}
) where {T}

    # the internal  GPUsolver is type unstable, so multiple
    # calls to the  GPUsolvers will be very slow if called
    # directly.   Grab it here and then call an inner function
    # so that the  GPUsolver has concrete type
    GPUsolver = kktsolver.GPUsolver
    return _kktsolver_update_inner!(kktsolver,GPUsolver,cones)
end


function _kktsolver_update_inner!(
    kktsolver:: GPULDLKKTSolver{T},
    GPUsolver::AbstractDirectLDLSolver{T},
    cones::CompositeConeGPU{T}
) where {T}

    #real implementation is here, and now  GPUsolver
    #will be compiled to something concrete.

    map       = kktsolver.map
    KKT       = kktsolver.KKT

    #Set the elements the W^tW blocks in the KKT matrix.
    get_Hs!(cones,kktsolver.Hsblocks)

    @. kktsolver.Hsblocks *= -one(T)
    _update_values!(GPUsolver,KKT,map.Hsblocks,kktsolver.Hsblocks)

    return _kktsolver_regularize_and_refactor!(kktsolver, GPUsolver)

end

function _kktsolver_regularize_and_refactor!(
    kktsolver::GPULDLKKTSolver{T},
    GPUsolver::AbstractDirectLDLSolver{T}
) where{T}

    settings      = kktsolver.settings
    map           = kktsolver.map
    KKT        = kktsolver.KKT
    Dsigns        = kktsolver.Dsigns
    diag_kkt      = kktsolver.work1
    diag_shifted  = kktsolver.work2


    if(settings.static_regularization_enable)

        # hold a copy of the true KKT diagonal
        @views diag_kkt .= KKT.nzVal[map.diag_full]
        ϵ = _compute_regularizer(diag_kkt, settings)

        # compute an offset version, accounting for signs
        diag_shifted .= diag_kkt

        diag_shifted .+= Dsigns*ϵ

        # overwrite the diagonal of KKT and within the  GPUsolver
        _update_diag_values_KKT!(KKT,map.diag_full,diag_shifted)

        # remember the value we used.  Not needed,
        # but possibly useful for debugging
        kktsolver.diagonal_regularizer = ϵ

    end

    is_success = refactor!(GPUsolver)

    if(settings.static_regularization_enable)

        # put our internal copy of the KKT matrix back the way
        # it was. Not necessary to fix the  GPUsolver copy because
        # this is only needed for our post-factorization IR scheme

        _update_diag_values_KKT!(KKT,map.diag_full,diag_kkt)

    end

    return is_success
end


function kktsolver_setrhs!(
    kktsolver::GPULDLKKTSolver{T},
    rhsx::AbstractVector{T},
    rhsz::AbstractVector{T}
) where {T}

    b = kktsolver.b
    (m,n) = (kktsolver.m,kktsolver.n)

    b[1:n]             .= rhsx
    b[(n+1):(n+m)]     .= rhsz
    
    CUDA.synchronize()

    return nothing
end


function kktsolver_getlhs!(
    kktsolver::GPULDLKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    x = kktsolver.x
    (m,n) = (kktsolver.m,kktsolver.n)

    isnothing(lhsx) || (@views lhsx .= x[1:n])
    isnothing(lhsz) || (@views lhsz .= x[(n+1):(n+m)])

    CUDA.synchronize()

    return nothing
end


function kktsolver_solve!(
    kktsolver::GPULDLKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    (x,b) = (kktsolver.x,kktsolver.b)

    solve!(kktsolver.GPUsolver,x,b)

    is_success = begin
        if(kktsolver.settings.iterative_refinement_enable)
            #IR reports success based on finite normed residual
            is_success = _iterative_refinement(kktsolver,kktsolver.GPUsolver)
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

# update methods for P and A 
function kktsolver_update_P!(
    kktsolver::GPULDLKKTSolver{T},
    P::CuSparseMatrix{T}
) where{T}
    _update_values!(kktsolver.GPUsolver,kktsolver.KKT,kktsolver.map.P,P.nzVal)
end

function kktsolver_update_A!(
    kktsolver::GPULDLKKTSolver{T},
    A::CuSparseMatrix{T}
) where{T}
    _update_values!(kktsolver.GPUsolver,kktsolver.KKT,kktsolver.map.A,A.nzVal)
end

function kktsolver_update_At!(
    kktsolver::GPULDLKKTSolver{T},
    At::CuSparseMatrix{T}
) where{T}
    _update_values!(kktsolver.GPUsolver,kktsolver.KKT,kktsolver.map.At,At.nzVal)
end

function  _iterative_refinement(
    kktsolver::GPULDLKKTSolver{T},
    GPUsolver::AbstractDirectLDLSolver{T}
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

    for i = 1:IR_maxiter

        if(norme <= IR_abstol + IR_reltol*normb)
            # within tolerance, or failed.  Exit
            break
        end
        lastnorme = norme

        #make a refinement and continue
        solve!(GPUsolver,dx,e)

        #prospective solution is x + dx.   Use dx space to
        #hold it for a check before applying to x
        @. dx += x
        CUDA.synchronize()
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
    KKT::AbstractCuSparseMatrix{T},
    ξ::AbstractVector{T}) where {T}

    
    mul!(e,KKT,ξ)    # e = b - Kξ
    @. e = b - e
    CUDA.synchronize()
    norme = norm(e,Inf)

    return norme
end
