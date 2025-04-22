using SparseArrays, StaticArrays

function _allocate_kkt_Hsblocks_gpu(
    ::Type{Z}, 
    cones::CompositeConeGPU{T}
) where{T <: AbstractFloat, Z <: Real}

    rng_blocks = cones.rng_blocks
    CUDA.@allowscalar nnz = length(rng_blocks) == 0 ? 0 : last(rng_blocks[end])
    CUDA.zeros(Z,nnz)

end

struct GPUDataMap

    P::CuVector{Cint}
    A::CuVector{Cint}
    At::CuVector{Cint}        #YC: not sure whether we need it or not
    Hsblocks::CuVector{Cint}                #indices of the lower RHS blocks (by cone)

    #all of above terms should be disjoint and their union
    #should cover all of the user data in the KKT matrix.  Now
    #we make two last redundant indices that will tell us where
    #the whole diagonal is, including structural zeros.
    diagP::CuVector{Cint}
    diag_full::CuVector{Cint}

    #mapping for sparse SOCs 
    u::CuVector{Cint}
    v::CuVector{Cint}
    ut::CuVector{Cint}
    vt::CuVector{Cint}
    D::CuVector{Cint}
    
    function GPUDataMap(
        Pmat::CuSparseMatrix{T},
        Amat::CuSparseMatrix{T},
        cones::CompositeConeGPU{T}
        ) where{T}

        (m,n) = (size(Amat,1), size(Pmat,1))
        P = CUDA.zeros(Cint,nnz(Pmat))
        A = CUDA.zeros(Cint,nnz(Amat))
        At = CUDA.zeros(Cint,nnz(Amat))

        #the diagonal of the ULHS block P.
        #NB : we fill in structural zeros here even if the matrix
        #P is empty (e.g. as in an LP), so we can have entries in
        #index Pdiag that are not present in the index P
        diagP  = CUDA.zeros(Cint,n)

        #make an index for each of the Hs blocks for each cone
        Hsblocks = _allocate_kkt_Hsblocks_gpu(Cint, cones)

        #now do the sparse cone expansion pieces
        n_linear = cones.n_linear
        n_sparse_soc = cones.n_sparse_soc
        CUDA.@allowscalar lensparse = sum(cone -> length(cone), cones.rng_cones[(n_linear+1):(n_linear+n_sparse_soc)])
        u = CUDA.zeros(Cint, lensparse)
        v = CUDA.zeros(Cint, lensparse)
        ut = CUDA.zeros(Cint, lensparse)
        vt = CUDA.zeros(Cint, lensparse)
        D = CUDA.zeros(Cint, 2*n_sparse_soc)

        diag_full = CUDA.zeros(Cint,m+n+2*n_sparse_soc)

        return new(P, A, At, Hsblocks, diagP, diag_full,
                    u, v, ut, vt, D)
    end

end

function _assemble_full_kkt_matrix(
    P::CuSparseMatrix{T},
    A::CuSparseMatrix{T},
    At::CuSparseMatrix{T},
    cones::CompositeConeGPU{T}
) where{T}
    map   = GPUDataMap(P,A,cones)
    (m,n) = (size(A,1), size(P,1))
    p     = length(map.D)

    #entries actually on the diagonal of P
    nnz_diagP  = _count_diagonal_entries_full_gpu(P)

    # total entries in the Hs blocks
    nnz_Hsblocks = length(map.Hsblocks)

    nnzKKT = (nnz(P) +          # Number of elements in P
    (n - nnz_diagP) +           # Number of elements in diagonal top left block, remove double count on the diagonal if P has entries
    2*nnz(A) +                  # Number of nonzeros in A and A'
    nnz_Hsblocks +              # Number of elements in diagonal below A'
    4*length(map.u) +           # Number of elements in sparse cone off diagonals
    p                           # Number of elements in diagonal of sparse cones
    )

    rowptr, colval, nzval = _csr_spalloc_gpu(T, Cint(m+n+p), Cint(nnzKKT))

    P_zero = CUDA.zeros(Cint, length(rowptr))
    fill!(P_zero, 0)
    # YC: Implementation is slightly different from CPU counterparts.
    #     On GPU, we store nnz items of each row at index i+1 rather than i to use function accumulate!().
    _full_kkt_assemble_colcounts_gpu(rowptr, P, P_zero, A, At, cones)     
    _full_kkt_assemble_fill_gpu(rowptr, colval, nzval, P, P_zero, A, At, cones, map)

    K = CuSparseMatrixCSR{T}(rowptr, colval, nzval, (m+n+p, m+n+p))
    P_zero = nothing    #release memory of P_zero

    return K,map

end

function _full_kkt_assemble_colcounts_gpu(
    rowptr::CuVector{Cint}, 
    P::CuSparseMatrix{T},
    P_zero::CuVector{Cint},
    A::CuSparseMatrix{T},
    At::CuSparseMatrix{T},
    cones::CompositeConeGPU{T}
) where{T}

    (m,n) = size(A)

    #use K.p to hold nnz entries in each
    #column of the KKT matrix
    fill!(rowptr, 0)

    #Count P, A, At of KKT
    _csr_rowcount_block_full_gpu(rowptr, P, At)
    _csr_rowcount_missing_diag_full_gpu(rowptr, P, P_zero)
    _csr_rowcount_block_gpu(rowptr, A, n)

    #Count the Hessian part of KKT
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones

    #Count the diagonal Hessian parts
    CUDA.@allowscalar begin
        for i in cones.idx_eq
            rng_cone_i = rng_cones[i] .+ (n + 1) 
            @views rowptr[rng_cone_i] .+= 1
        end
        for i in cones.idx_inq
            rng_cone_i = rng_cones[i] .+ (n + 1) 
            @views rowptr[rng_cone_i] .+= 1
        end
        #Assume the sparse socs is after linear cones and before dense socs
        @inbounds for i in (n_linear+1):(n_linear+n_sparse_soc)
            rng_cone_i = rng_cones[i] .+ (n + 1) 
            @views rowptr[rng_cone_i] .+= 1
        end            
    end
    CUDA.synchronize()

    #Count the additional sparse mapping for SOCs
    # track the next sparse row to fill 
    prow = m + n + 1 #next sparse row to fill
    
    n_sparse_soc = cones.n_sparse_soc
    CUDA.@allowscalar for i in (n_linear+1):(n_linear+n_sparse_soc)
    
        #add sparse expansions rows for sparse socs  
        rng_cone_i = rng_cones[i] .+ (n + 1)
        rowptr[prow+1] = length(rng_cone_i) + 1
        rowptr[prow+2] = length(rng_cone_i) + 1
        prow += 2 #next sparse row to fill 

        #add sparse expansions columns for sparse cones 
        @views rowptr[rng_cone_i] .+= 2
    end
    CUDA.synchronize()
    
    #Count the remaining dense block
    n_rec = (n_soc - n_sparse_soc) + n_exp + n_pow + n_psd
    if (n_rec > 0)
        _csr_rowcount_dense_full_gpu(rowptr, rng_cones, Cint(n), n_linear+n_sparse_soc, n_rec)
    end

    return nothing
end


function _full_kkt_assemble_fill_gpu(
    rowptr::CuVector{Cint}, 
    colval::CuVector{Cint},
    nzval::CuVector{T},
    P::CuSparseMatrix{T},
    P_zero::CuVector{Cint},
    A::CuSparseMatrix{T},
    At::CuSparseMatrix{T},
    cones::CompositeConeGPU{T},
    map::GPUDataMap
) where{T}

    (m,n) = size(A)
    n = Cint(n)

    #cumsum total entries to convert to K.p
    _csr_rowcount_to_rowptr_gpu(rowptr)

    #filling [P At;A 0] parts
    _csr_fill_P_diag_full_gpu(rowptr, colval, nzval, P, P_zero, map.P)
    _csr_fill_block_gpu(rowptr, colval, nzval, At, map.At, one(Cint), Cint(n+1))
    _csr_fill_block_gpu(rowptr, colval, nzval, A, map.A, Cint(n+1), one(Cint))

    #filling Hessian blocks for cones
    rng_cones = cones.rng_cones
    rng_blocks = cones.rng_blocks
    CUDA.@allowscalar begin
        for i in cones.idx_eq
            row = first(rng_cones[i]) + n
            block = view(map.Hsblocks,rng_blocks[i])
            _csr_fill_diag_gpu(rowptr, colval, nzval, block, row, Cint(length(block)))
        end
        for i in cones.idx_inq
            row = first(rng_cones[i]) + n
            block = view(map.Hsblocks,rng_blocks[i])
            _csr_fill_diag_gpu(rowptr, colval, nzval, block, row, Cint(length(block)))
        end
        #Assume the sparse socs is after linear cones and before dense socs
        n_linear = cones.n_linear
        n_sparse_soc = cones.n_sparse_soc
        @inbounds for i in (n_linear+1):(n_linear+n_sparse_soc)
            row = first(rng_cones[i]) + n
            block = view(map.Hsblocks,rng_blocks[i])
            _csr_fill_diag_gpu(rowptr, colval, nzval, block, row, Cint(length(block)))
        end
    end
    CUDA.synchronize()

    # Initializing additional rows (columns) for sparse second-order cones
    # track the next sparse row to fill
    prow = Cint(m + n + 1) #next sparse row to fill

    n_sparse_soc = cones.n_sparse_soc
    n_linear = cones.n_linear
    sparse_idx = 0

    CUDA.@allowscalar for i in 1:n_sparse_soc

        #add sparse expansions columns for sparse cones 
        rng_cone_i = rng_cones[n_linear + i] .+ n
        len_i = length(rng_cone_i)
        rng_vec_i = (sparse_idx+1):(sparse_idx+len_i)

        rowptr_i = view(rowptr, rng_cone_i)

        @views @. map.v[rng_vec_i] = rowptr_i
        @views @. colval[rowptr_i] = prow
        @views @. rowptr_i += 1

        @views @. map.u[rng_vec_i] = rowptr_i
        @views @. colval[rowptr_i] = prow + 1
        @views @. rowptr_i += 1

        #add sparse expansions rows for sparse cones 
        colidx = Cint(rng_cone_i.start - 1)

        rowidx = Cint(rowptr[prow] - 1)
        vti = view(map.vt, rng_vec_i)
        _csr_fill_sparsesoc_gpu(vti, colval, colidx, rowidx)
        rowptr[prow] += len_i

        rowidx = Cint(rowptr[prow+1] - 1)
        uti = view(map.ut, rng_vec_i)
        _csr_fill_sparsesoc_gpu(uti, colval, colidx, rowidx)
        rowptr[prow+1] += len_i

        sparse_idx += len_i
        prow += 2 #next sparse row to fill 
    end

    if n_sparse_soc > 0
        #final ssttings for map.D 
        rng_D = (m + n + 1):(m + n + 2*n_sparse_soc)
        rowptr_D = view(rowptr, rng_D)
        CUDA.@sync @. map.D = rowptr_D
        _fill_range_gpu(map.D, colval, Cint(m+n))
        CUDA.@sync @. rowptr_D += 1

        #fill in nzval with 0s
        @views @. nzval[map.u] = 0.
        @views @. nzval[map.v] = 0.
        @views @. nzval[map.ut] = 0.
        @views @. nzval[map.vt] = 0.
        @views @. nzval[map.D] = 0.
    end
    CUDA.synchronize()

    n_linear = cones.n_linear
    n_rec = (cones.n_soc - n_sparse_soc) + cones.n_exp + cones.n_pow + cones.n_psd
    if (n_rec > 0)
        _csr_fill_dense_full_gpu(rowptr, colval, nzval, map.Hsblocks, rng_cones, rng_blocks, n_linear+n_sparse_soc, Cint(n), n_rec)
    end
    
    #backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs_gpu(rowptr)

    #Now we can populate the index of the full diagonal.
    #We have filled in structural zeros on it everywhere.

    _map_diag_full_gpu(rowptr, colval, map.diag_full)
    CUDA.@sync @views map.diagP     .= map.diag_full[1:n]

    return nothing
end