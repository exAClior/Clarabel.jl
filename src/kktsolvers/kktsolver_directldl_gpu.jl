# -------------------------------------
# KKTSolver using cudss direct solvers
# -------------------------------------

const CuVectorView{T} = SubArray{T, 1, AbstractVector{T}, Tuple{AbstractVector{Int}}, false}
##############################################################
# YC: Some functions are repeated as in the direct solver, which are better to be removed
##############################################################
mutable struct GPULDLKKTSolver{T} <: AbstractKKTSolver{T}

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
    mapcpu::IndirectDataMap
    mapgpu::GPUDataMap 

    #the expected signs of D in KKT = LDL^T
    Dsigns::AbstractVector{Int}

    # a vector for storing the Hs blocks
    # on the in the KKT matrix block diagonal
    Hsblocks::AbstractVector{T}

    #unpermuted KKT matrix
    KKTcpu::SparseMatrixCSC{T,Int}
    KKTgpu::AbstractCuSparseMatrix{T}

    #settings just points back to the main solver settings.
    #Required since there is no separate LDL settings container
    settings::Settings{T}

    #the direct linear LDL solver
    GPUsolver::AbstractGPUSolver{T}

    #the diagonal regularizer currently applied
    diagonal_regularizer::T

    function GPULDLKKTSolver{T}(P,A,cones,m,n,settings) where {T}

        # get a constructor for the LDL solver we should use,
        # and also the matrix shape it requires
        (kktshape, GPUsolverT) = _get_GPUsolver_config(settings)

        #construct a KKT matrix of the right shape
        KKTcpu, mapcpu = _assemble_full_kkt_matrix(P,A,cones,kktshape)
        KKTgpu = CuSparseMatrixCSR(KKTcpu)

        #YC: update GPU map, should be removed later on
        mapgpu   = GPUDataMap(P,A,cones,mapcpu)

        #YC: disabled sparse expansion
        #Need this many extra variables for sparse cones
        p = pdim(mapcpu.sparse_maps)
        if p > 0 
            error("We don't support the augmented sparse cone at the moment!")
        end

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        dim = m + n + p

        #LHS/RHS/work for iterative refinement
        x    = CuVector{T}(undef,dim)
        b    = CuVector{T}(undef,dim)
        work1 = CuVector{T}(undef,dim)
        work2 = CuVector{T}(undef,dim)

        #the expected signs of D in LDL
        Dsigns_cpu = Vector{Int}(undef,dim)
        _fill_Dsigns!(Dsigns_cpu,m,n,mapcpu)       #This is run on CPU
        Dsigns = CuVector(Dsigns_cpu)

        Hsblocks = CuVector(_allocate_kkt_Hsblocks(T, cones))

        diagonal_regularizer = zero(T)

        #the indirect linear solver engine
        GPUsolver = GPUsolverT{T}(KKTgpu,A,settings)

        return new(m,n,p,x,b,
                   work1,work2,mapcpu,mapgpu,
                   Dsigns,
                   Hsblocks,
                   KKTcpu,KKTgpu,settings,GPUsolver,
                   diagonal_regularizer
                   )
    end

end

GPULDLKKTSolver(args...) = GPULDLKKTSolver{DefaultFloat}(args...)

function _get_GPUsolver_type(s::Symbol)
    try
        return GPUSolversDict[s]
    catch
        throw(error("Unsupported gpu linear solver :", s))
    end
end

function _get_GPUsolver_config(settings::Settings)

    #which LDL solver should I use?
    GPUsolverT = _get_GPUsolver_type(settings.direct_solve_method)

    #does it want a :full KKT matrix?
    kktshape = required_matrix_shape(GPUsolverT)
    @assert(kktshape == :full)

    (kktshape,GPUsolverT)
end 


#update entries in the kktsolver object using the
#given index into its CSC representation
function _update_values!(
    GPUsolver::AbstractGPUSolver{T},
    KKT::AbstractSparseMatrix{T},
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


# #scale entries in the kktsolver object using the
# #given index into its CSC representation
# function _scale_values!(
#     GPUsolver::AbstractGPUSolver{T},
#     KKT::SparseMatrixCSC{T,Ti},
#     index::AbstractVector{Ti},
#     scale::T
# ) where{T,Ti}

#     #Update values in the KKT matrix K
#     @. KKT.nzval[index] *= scale

# end


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
    GPUsolver::AbstractGPUSolver{T},
    cones::CompositeConeGPU{T}
) where {T}

    #real implementation is here, and now  GPUsolver
    #will be compiled to something concrete.

    map       = kktsolver.mapgpu
    KKT       = kktsolver.KKTgpu

    #Set the elements the W^tW blocks in the KKT matrix.
    # get_Hs!(cones,kktsolver.Hsblockscpu,false)
    get_Hs!(cones,kktsolver.Hsblocks)

    @. kktsolver.Hsblocks *= -one(T)
    _update_values!(GPUsolver,KKT,map.Hsblocks,kktsolver.Hsblocks)

    #YC: disabled sparse expansion
    # sparse_map_iter = Iterators.Stateful(map.sparse_maps)

    # updateFcn = (index,values) -> _update_values!(GPUsolver,KKTcpu,index,values)
    # scaleFcn  = (index,scale)  -> _scale_values!(GPUsolver,KKTcpu,index,scale)

    # for cone in cones
    #     #add sparse expansions columns for sparse cones 
    #     if @conedispatch is_sparse_expandable(cone)
    #         thismap = popfirst!(sparse_map_iter)
    #         _csc_update_sparsecone_full(cone,thismap,updateFcn,scaleFcn)
    #     end 
    # end

    return _kktsolver_regularize_and_refactor!(kktsolver, GPUsolver)

end

function _kktsolver_regularize_and_refactor!(
    kktsolver::GPULDLKKTSolver{T},
    GPUsolver::AbstractGPUSolver{T}
) where{T}

    settings      = kktsolver.settings
    map           = kktsolver.mapgpu
    KKTgpu        = kktsolver.KKTgpu
    Dsigns        = kktsolver.Dsigns
    diag_kkt      = kktsolver.work1
    diag_shifted  = kktsolver.work2


    if(settings.static_regularization_enable)

        # hold a copy of the true KKT diagonal
        @views diag_kkt .= KKTgpu.nzVal[map.diag_full]
        ϵ = _compute_regularizer(diag_kkt, settings)

        # compute an offset version, accounting for signs
        diag_shifted .= diag_kkt

        diag_shifted .+= Dsigns*ϵ

        # overwrite the diagonal of KKT and within the  GPUsolver
        _update_diag_values_KKT!(KKTgpu,map.diag_full,diag_shifted)

        # remember the value we used.  Not needed,
        # but possibly useful for debugging
        kktsolver.diagonal_regularizer = ϵ

    end

    is_success = refactor!(GPUsolver)

    if(settings.static_regularization_enable)

        # put our internal copy of the KKT matrix back the way
        # it was. Not necessary to fix the  GPUsolver copy because
        # this is only needed for our post-factorization IR scheme

        _update_diag_values_KKT!(KKTgpu,map.diag_full,diag_kkt)

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
    kktsolver::GPULDLKKTSolver{T},
    rhsx::AbstractVector{T},
    rhsz::AbstractVector{T}
) where {T}

    b = kktsolver.b
    (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)

    b[1:n]             .= rhsx
    b[(n+1):(n+m)]     .= rhsz
    b[(n+m+1):(n+m+p)] .= 0
    
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

function  _iterative_refinement(
    kktsolver::GPULDLKKTSolver{T},
    GPUsolver::AbstractGPUSolver{T}
) where{T}

    (x,b)   = (kktsolver.x,kktsolver.b)
    (e,dx)  = (kktsolver.work1, kktsolver.work2)
    settings = kktsolver.settings

    #iterative refinement params
    IR_reltol    = settings.iterative_refinement_reltol
    IR_abstol    = settings.iterative_refinement_abstol
    IR_maxiter   = settings.iterative_refinement_max_iter
    IR_stopratio = settings.iterative_refinement_stop_ratio

    KKT = kktsolver.KKTgpu
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
