# -------------------------------------
# KKTSolver using cudss direct solvers
# -------------------------------------

const CuVectorView{T} = SubArray{T, 1, CuVector{T}, Tuple{CuVector{Int}}, false}
##############################################################
# YC: Some functions are repeated as in the direct solver, which are better to be removed
##############################################################
mutable struct GPULDLKKTSolver{T} <: AbstractKKTSolver{T}

    # problem dimensions
    m::Int; n::Int; p::Int;

    # Left and right hand sides for solves
    x::CuVector{T}
    b::CuVector{T}

    # internal workspace for IR scheme
    # and static offsetting of KKT
    work1::CuVector{T}
    work2::CuVector{T}

    #KKT mapping from problem data to KKT
    map::GPUDataMap 

    cones::CompositeConeGPU{T}
    P::CuSparseMatrix{T}
    A::CuSparseMatrix{T}
    At::CuSparseMatrix{T}

    #the expected signs of D in KKT = LDL^T
    Dsigns::CuVector{Cint}

    # a vector for storing the Hs blocks
    # on the in the KKT matrix block diagonal
    Hsblocks::CuVector{T}

    #unpermuted KKT matrix
    KKT::CuSparseMatrix{T}

    #settings just points back to the main solver settings.
    #Required since there is no separate LDL settings container
    settings::Settings{T}

    #the direct linear LDL solver
    GPUsolver::AbstractDirectLDLSolver{T}

    #the diagonal regularizer currently applied
    diagonal_regularizer::T

    function GPULDLKKTSolver{T}(
        P::CuSparseMatrix{T},
        A::CuSparseMatrix{T},
        At::CuSparseMatrix{T},
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

        #Need this many extra variables for sparse cones
        p = length(map.D)

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        dim = m + n + p

        #LHS/RHS/work for iterative refinement
        x    = CuVector{T}(undef,dim)
        b    = CuVector{T}(undef,dim)
        work1 = CuVector{T}(undef,dim)
        work2 = CuVector{T}(undef,dim)

        #the expected signs of D in LDL
        Dsigns = CUDA.ones(Cint,dim)
        @views fill!(Dsigns[n+1:n+m], -one(Cint))
        @views fill!(Dsigns[n+m+1:2:n+m+p], -one(Cint))     #for sparse SOCs

        Hsblocks = _allocate_kkt_Hsblocks_gpu(T, cones)
        diagonal_regularizer = zero(T)

        #the indirect linear solver engine
        GPUsolver = GPUsolverT{T}(KKT,x,b)

        return new(m,n,p,x,b,
                   work1,work2,map,
                   cones,P,A,At,
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
    index::CuVector{Ti},
    values::CuVector{T}
) where{T,Ti}

    #Update values in the KKT matrix K
    CUDA.@sync @. KKT.nzVal[index] = values

end

@inline function _update_value!(
    KKT::AbstractMatrix{T},
    index::Ti,
    value::T
) where{T,Ti}

    #Update values in the KKT matrix K
    KKT.nzVal[index] = value

end

@inline function _update_value!(
    nzVal::AbstractVector{T},
    index::Ti,
    value::T
) where{T,Ti}

    #Update values in the KKT matrix K
    nzVal[index] = value

end

function _scaled_update_values!(
    GPUsolver::AbstractDirectLDLSolver{T},
    KKT::CuSparseMatrix{T},
    index::CuVector{Ti},
    values::CuVector{T},
    scale::T
) where{T,Ti}

    #Update values in the KKT matrix K
    @. KKT.nzVal[index] = scale*values

end

#updates KKT matrix values
function _update_diag_values_KKT!(
    KKT::CuSparseMatrix{T},
    index::CuVector{Ti},
    values::CuVector{T}
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

    CUDA.@sync @. kktsolver.Hsblocks *= -one(T)
    _update_values!(GPUsolver,KKT,map.Hsblocks,kktsolver.Hsblocks)

    n_shift = cones.n_linear
    n_sparse_soc = cones.n_sparse_soc
    numel_linear = cones.numel_linear
    rng_cones = cones.rng_cones
    η = cones.η

    # Update for the KKT part of the sparse socs
    if n_sparse_soc > SPARSE_SOC_PARALELL_NUM
        update_KKT_sparse_soc_parallel!(KKT.nzVal, η, map.D, map.vu, map.vut, cones.vut, rng_cones, numel_linear, n_shift, n_sparse_soc)
    elseif n_sparse_soc > 0
        update_KKT_sparse_soc_sequential!(GPUsolver, KKT, map, cones)
    end

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
        CUDA.@sync @views diag_kkt .= KKT.nzVal[map.diag_full]
        ϵ = _compute_regularizer(diag_kkt, settings)

        # compute an offset version, accounting for signs
        CUDA.@sync @. diag_shifted = diag_kkt + Dsigns*ϵ

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
    rhsx::CuVector{T},
    rhsz::CuVector{T}
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
    lhsx::Union{Nothing,CuVector{T}},
    lhsz::Union{Nothing,CuVector{T}}
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
    lhsx::Union{Nothing,CuVector{T}},
    lhsz::Union{Nothing,CuVector{T}}
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
    # norme = _get_refine_error!(kktsolver,e,b,x) 
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
    e::CuVector{T},
    b::CuVector{T},
    KKT::CuSparseMatrix{T},
    ξ::CuVector{T}) where {T}

    
    mul!(e,KKT,ξ)    # e = b - Kξ
    @. e = b - e
    CUDA.synchronize()
    norme = norm(e,Inf)

    return norme
end

# function _get_refine_error!(
#     kktsolver::GPULDLKKTSolver{T},
#     e::CuVector{T},
#     b::CuVector{T},
#     ξ::CuVector{T}) where {T}

#     m = kktsolver.m 
#     n = kktsolver.n
#     p = kktsolver.p

#     cones = kktsolver.cones
#     Matvut = cones.Matvut
#     P = kktsolver.P
#     A = kktsolver.A
#     At = kktsolver.At

#     e1 = view(e, 1:n)
#     e2 = view(e, (n+1):(n+m))
#     ξ1 = view(ξ, 1:n)
#     ξ2 = view(ξ, (n+1):(n+m))
    
#     # mul!(e,KKT,ξ)    # e = b - Kξ
#     mul!(e1, At, ξ2)
#     mul!(e1, P, ξ1, one(T), one(T))
#     CUDA.@sync @. e2 = 0
#     Clarabel.mul_Hs_diag!(cones, e2, ξ2)
#     mul!(e2, A, ξ1, one(T), -one(T))

#     n_sparse_soc = cones.n_sparse_soc
#     d = cones.d
#     η = cones.η

#     CUDA.@allowscalar if n_sparse_soc > 0
#         e3 = view(e, (n+m+1):(n+m+p))
#         ξ3 = view(ξ, (n+m+1):(n+m+p))
#         @inbounds for i in 1:n_sparse_soc
#             shift_i = i + cones.n_linear
#             rng_i = cones.rng_cones[shift_i]
#             rng_sparse_i = rng_i .- cones.numel_linear
#             e2i = view(e2, rng_i)
#             ξ2i = view(ξ2, rng_i)
#             η2 = η[i]^2

#             e2i[1] -= η2*(d[i] - 1)*ξ2i[1]
#             CUDA.@sync @. e2i -= η2*ξ2i
#             e3[2*i-1] = -η2*ξ3[2*i-1]
#             e3[2*i] = η2*ξ3[2*i]
#         end

#         mul!(e2, Matvut', ξ3, one(T), one(T))
#         mul!(e3, Matvut, ξ2, one(T), one(T))
#     end

#     CUDA.@sync @. e = b - e
#     # println(e[(n+1):(n+2)])
#     norme = norm(e,Inf)

#     return norme
# end

# function _get_refine_error!(
#     kktsolver::GPULDLKKTSolver{T},
#     e::CuVector{T},
#     b::CuVector{T},
#     ξ::CuVector{T}) where {T}

#     m = kktsolver.m 
#     n = kktsolver.n
#     p = kktsolver.p

#     cones = kktsolver.cones
#     P = kktsolver.P
#     A = kktsolver.A
#     At = kktsolver.At

#     e1 = view(e, 1:n)
#     e2 = view(e, (n+1):(n+m))
#     ξ1 = view(ξ, 1:n)
#     ξ2 = view(ξ, (n+1):(n+m))
    
#     # mul!(e,KKT,ξ)    # e = b - Kξ
#     mul!(e1, At, ξ2)
#     CUDA.@sync @. e2 = 0
#     Clarabel.mul_Hs_diag!(cones, e2, ξ2)
#     mul!(e1, P, ξ1, one(T), one(T))
#     mul!(e2, A, ξ1, one(T), -one(T))

#     n_sparse_soc = cones.n_sparse_soc
#     d = cones.d
#     u = cones.u
#     v = cones.v
#     η = cones.η

#     CUDA.@allowscalar if n_sparse_soc > 0
#         e3 = view(e, (n+m+1):(n+m+p))
#         ξ3 = view(ξ, (n+m+1):(n+m+p))
#         @inbounds for i in 1:n_sparse_soc
#             shift_i = i + cones.n_linear
#             rng_i = cones.rng_cones[shift_i]
#             rng_sparse_i = rng_i .- cones.numel_linear
#             ui = view(u, rng_sparse_i)
#             vi = view(v, rng_sparse_i)
#             e2i = view(e2, rng_i)
#             ξ2i = view(ξ2, rng_i)
#             η2 = η[i]^2

#             e2i[1] -= η2*(d[i] - 1)*ξ2i[1]
#             CUDA.@sync @. e2i -= η2*(ξ2i + vi*ξ3[2*i-1] + ui*ξ3[2*i])
#             e3[2*i-1] = -η2*(dot(ξ2i, vi) + ξ3[2*i-1])
#             e3[2*i] = -η2*(dot(ξ2i, ui) - ξ3[2*i])
#         end
#     end
#     CUDA.synchronize()
#     @. e = b - e
#     CUDA.synchronize()
#     # println(e[(n+1):(n+2)])
#     norme = norm(e,Inf)

#     return norme
# end

#Parallel update for KKT of the sparse socs
function _kernel_update_KKT_sparse_soc_parallel!(
    nzVal::AbstractVector{T},
    η::AbstractVector{T},
    D::AbstractVector{Cint},
    vu::AbstractVector{Cint},
    vut::AbstractVector{Cint},
    vutval::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_sparse_soc
        η2 = η[i]*η[i]

        shift_i = i + n_shift
        rng_i = rng_cones[shift_i] .- numel_linear
        endidx = 2*rng_i.stop
        startidx = endidx - 2*length(rng_i) + 1

        @inbounds for j in startidx:endidx
            val = -η2*vutval[j]
            nzVal[vu[j]] = val
            nzVal[vut[j]] = val
        end
    
        #set diagonal to η^2*(-1,1) in the extended rows/cols
        nzVal[D[2*i-1]] = -η2
        nzVal[D[2*i]] = η2
    end

    return nothing
end

@inline function update_KKT_sparse_soc_parallel!(
    nzVal::AbstractVector{T},
    η::AbstractVector{T},
    D::AbstractVector{Cint},
    vu::AbstractVector{Cint},
    vut::AbstractVector{Cint},
    vutval::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    
    kernel = @cuda launch=false _kernel_update_KKT_sparse_soc_parallel!(nzVal, η, D, vu, vut, vutval, rng_cones, numel_linear, n_shift, n_sparse_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_sparse_soc, config.threads)
    blocks = cld(n_sparse_soc, threads)

    CUDA.@sync kernel(nzVal, η, D, vu, vut, vutval, rng_cones, numel_linear, n_shift, n_sparse_soc; threads, blocks)

end

#Sequential update for KKT of the sparse socs
@inline function update_KKT_sparse_soc_sequential!(
    GPUsolver::AbstractDirectLDLSolver{T},
    KKT::CuSparseMatrix{T},
    map::GPUDataMap,
    cones::CompositeConeGPU{T}
) where {T}
    
    n_shift = cones.n_linear

    sparse_idx = 0
    prow = one(Cint)
    CUDA.@allowscalar for i in 1:cones.n_sparse_soc
        ηi = cones.η[i]
        η2 = ηi*ηi

        shift_i = i + n_shift
        len_i = length(cones.rng_cones[shift_i])
        rng_i = (sparse_idx+1):(sparse_idx+2*len_i)

        vu_idx = view(map.vu, rng_i)
        vut_idx = view(map.vut, rng_i)
        vut_val = view(cones.vut, rng_i)

        _scaled_update_values!(GPUsolver,KKT,vu_idx,vut_val, -η2)
        _scaled_update_values!(GPUsolver,KKT,vut_idx,vut_val, -η2)

        #set diagonal to η^2*(-1,1) in the extended rows/cols
        _update_value!(KKT,map.D[prow], -η2)
        _update_value!(KKT,map.D[prow+1], η2)

        sparse_idx += 2*len_i
        prow += 2
    end

    CUDA.synchronize()
end