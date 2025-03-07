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

        diag_full = CUDA.zeros(Cint,m+n)

        return new(P, A, At, Hsblocks, diagP, diag_full)
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

    #entries actually on the diagonal of P
    nnz_diagP  = _count_diagonal_entries_full_gpu(P)

    # total entries in the Hs blocks
    nnz_Hsblocks = length(map.Hsblocks)

    nnzKKT = (nnz(P) +        # Number of elements in P
    (n - nnz_diagP) +         # Number of elements in diagonal top left block, remove double count on the diagonal if P has entries
    2*nnz(A) +                # Number of nonzeros in A and A'
    nnz_Hsblocks              # Number of elements in diagonal below A'
    )

    rowptr, colval, nzval = _csr_spalloc_gpu(T, Cint(m+n), Cint(nnzKKT))

    P_zero = CUDA.zeros(Cint, length(rowptr))
    fill!(P_zero, false)
    _full_kkt_assemble_colcounts_gpu(rowptr, P, P_zero, A, At, cones)       
    _full_kkt_assemble_fill_gpu(rowptr, colval, nzval, P, P_zero, A, At, cones, map)

    K = CuSparseMatrixCSR{T}(rowptr, colval, nzval, (m+n, m+n))
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

    #Count first n columns of KKT
    _csr_rowcount_block_full_gpu(rowptr, P, At)
    _csr_rowcount_missing_diag_full_gpu(rowptr, P, P_zero)
    _csr_rowcount_block_gpu(rowptr, A, n+1)

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones

    CUDA.@allowscalar begin
        for i in cones.idx_eq
            rng_cone_i = rng_cones[i] .+ n 
            @views rowptr[rng_cone_i] .+= 1
        end
        for i in cones.idx_inq
            rng_cone_i = rng_cones[i] .+ n 
            @views rowptr[rng_cone_i] .+= 1
        end
    end
    CUDA.synchronize()

    n_rec = n_soc + n_exp + n_pow + n_psd
    if (n_rec > 0)
        _csr_rowcount_dense_full_gpu(rowptr, rng_cones, Cint(n), n_linear, n_rec)
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
    n_linear = cones.n_linear
    n_rec = cones.n_soc + cones.n_exp + cones.n_pow + cones.n_psd
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
    end
    CUDA.synchronize()

    if (n_rec > 0)
        _csr_fill_dense_full_gpu(rowptr, colval, nzval, map.Hsblocks, rng_cones, rng_blocks, n_linear, Cint(n), n_rec)
    end
    
    #backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs_gpu(rowptr)

    #Now we can populate the index of the full diagonal.
    #We have filled in structural zeros on it everywhere.

    _map_diag_full_gpu(rowptr, colval, map.diag_full)
    CUDA.@sync @views map.diagP     .= map.diag_full[1:n]

    return nothing
end