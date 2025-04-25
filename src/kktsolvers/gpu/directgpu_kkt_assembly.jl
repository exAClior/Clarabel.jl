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

    #index mapping for sparse SOCs 
    vu::CuVector{Cint}
    vut::CuVector{Cint}
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
        vu = CUDA.zeros(Cint, 2*lensparse)
        vut = CUDA.zeros(Cint, 2*lensparse)
        D = CUDA.zeros(Cint, 2*n_sparse_soc)

        diag_full = CUDA.zeros(Cint,m+n+2*n_sparse_soc)

        return new(P, A, At, Hsblocks, diagP, diag_full,
                    vu, vut, D)
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
    2*length(map.vu) +           # Number of elements in sparse cone off diagonals
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
    CUDA.unsafe_free!(P_zero)   #release memory of P_zero

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

    (m,n) = Cint.(size(A))

    #use K.p to hold nnz entries in each
    #column of the KKT matrix
    fill!(rowptr, 0)

    #Count P, A, At of KKT
    _csr_rowcount_block_full_gpu(rowptr, P, At)
    _csr_rowcount_missing_diag_full_gpu(rowptr, P, P_zero)
    _csr_rowcount_block_gpu(rowptr, A, n)

    #Count the Hessian part of KKT
    _csr_rowcount_Hessian_gpu(cones, rowptr, m, n)

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

    (m,n) = Cint.(size(A))

    #cumsum total entries to convert to K.p
    _csr_rowcount_to_rowptr_gpu(rowptr)

    #filling [P At;A 0] parts
    _csr_fill_P_diag_full_gpu(rowptr, colval, nzval, P, P_zero, map.P)
    _csr_fill_block_gpu(rowptr, colval, nzval, At, map.At, one(Cint), n+one(Cint))
    _csr_fill_block_gpu(rowptr, colval, nzval, A, map.A, n+one(Cint), one(Cint))

    #filling Hessian blocks for cones
    _csr_fill_Hessian_gpu(cones, map, rowptr, colval, nzval, m, n)

    #backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs_gpu(rowptr)

    #Now we can populate the index of the full diagonal.
    #We have filled in structural zeros on it everywhere.

    _map_diag_full_gpu(rowptr, colval, map.diag_full)
    CUDA.@sync @views map.diagP     .= map.diag_full[1:n]

    return nothing
end