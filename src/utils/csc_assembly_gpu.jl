using SparseArrays

function _csr_spalloc_gpu(T::Type{<:AbstractFloat}, n::Cint, nnz::Cint)

    rowptr = CUDA.zeros(Cint,n+1)
    colval = CUDA.zeros(Cint,nnz)
    nzval  = CUDA.zeros(T,nnz)

    return rowptr, colval, nzval
end

# #increment the K.colptr by the number of nonzeros
# #in a dense upper/lower triangle on the diagonal.
# function _csr_colcount_dense_triangle(K,initcol,blockcols,shape)
#     cols  = initcol:(initcol + (blockcols - 1))
#     if shape === :triu
#         @views K.colptr[cols] += 1:blockcols
#     else
#         @views K.colptr[cols] += blockcols:-1:1
#     end
# end

#increment the K.colptr by the number of nonzeros
#in a dense block on the diagonal.
function _kernel_csr_rowcount_dense_full(
    rowptr::AbstractVector{Cint}, 
    rng_cones::AbstractVector, 
    row_shift::Cint, 
    n_shift::Cint,
    n_cones::Cint
)

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= n_cones
        rng_cones_i = rng_cones[n_shift+i] .+ (row_shift + 1)
        len_i = length(rng_cones_i)

        @inbounds for idx in rng_cones_i
            rowptr[idx] += len_i            #add blockdim for each row
        end
    end

    return nothing
end

@inline function _csr_rowcount_dense_full_gpu(
    rowptr::AbstractVector{Cint}, 
    rng_cones::AbstractVector, 
    row_shift::Cint, 
    n_shift::Cint,
    n_cones::Cint
)
    @assert(length(rng_cones) == n_shift+n_cones)

    kernel = @cuda launch=false _kernel_csr_rowcount_dense_full(rowptr, rng_cones, row_shift, n_shift, n_cones)
    config = launch_configuration(kernel.fun)
    threads = min(n_cones, config.threads)
    blocks = cld(n_cones, threads)

    CUDA.@sync kernel(rowptr, rng_cones, row_shift, n_shift, n_cones; threads, blocks)
end

# #same as _kkt_count_diag, but counts places
# #where the input matrix M has a missing
# #diagonal entry.  M must be square and TRIU
# function _csr_colcount_missing_diag(K,M,initcol)

#     for i = 1:M.n
#         if((M.colptr[i] == M.colptr[i+1]) ||    #completely empty column
#            (M.rowval[M.colptr[i+1]-1] != i)     #last element is not on diagonal
#           )
#             K.colptr[i + (initcol-1)] += 1
#         end
#     end
# end

function _kernel_csr_rowcount_missing_diag_full(
    rowptr::AbstractVector{Cint}, 
    P::AbstractSparseMatrix{T},
    P_zero::AbstractVector{Cint}
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= P.dims[1]
        if(P.rowPtr[i] == P.rowPtr[i+1])    #completely empty row
            rowptr[i+1] += 1
            P_zero[i] = -1
            return nothing
        end
        @views colVali = P.colVal[P.rowPtr[i]:P.rowPtr[i+1]-1]     
        #check whether the i-th diagonal is missing
        @inbounds for idx in colVali
            if (idx >= i) 
                if (idx != i)
                    #i-th diagonal is missing
                    rowptr[i+1] += 1
                end
                P_zero[i] = idx
                return nothing
            end
        end
    end

    return nothing
end

@inline function _csr_rowcount_missing_diag_full_gpu(
    rowptr::AbstractVector{Cint}, 
    P::AbstractSparseMatrix{T},
    P_zero::AbstractVector{Cint}
) where {T}

    kernel = @cuda launch=false _kernel_csr_rowcount_missing_diag_full(rowptr, P, P_zero)
    config = launch_configuration(kernel.fun)
    threads = min(P.dims[1], config.threads)
    blocks = cld(P.dims[1], threads)

    CUDA.@sync kernel(rowptr, P, P_zero; threads, blocks)
end

# function _check_missing_diag_full(arr,k)
#     for val in arr
#         if (val >= k) 
#             return val  #val != k implies missing diagonal
#         end
#     end

#     return 0      #missing diagonal and all values in arr is smaller than k
# end

# #increment the K.colptr by the a number of nonzeros.
# #used to account for the placement of a column
# #vector that partially populates the column
# function _csr_colcount_colvec(K,n,firstrow, firstcol)

#     #just add the vector length to this column
#     K.colptr[firstcol] += n

# end

# #increment the K.colptr by 1 for every element
# #used to account for the placement of a column
# #vector that partially populates the column
# function _csr_colcount_rowvec(K,n,firstrow,firstcol)

#     #add one element to each of n consective columns
#     #starting from initcol.  The row index doesn't
#     #matter here.
#     for i = 1:n
#         K.colptr[firstcol + i - 1] += 1
#     end

# end

#increment the rowptr by the number of nonzeros in A 
function _kernel_csr_rowcount_block(
    rowptr::AbstractVector{Cint}, 
    A::AbstractSparseMatrix{T},
    shift::Int64
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= A.dims[1]
        rowptr[shift + i + 1] += A.rowPtr[i+1] - A.rowPtr[i]
    end

    return nothing
end

@inline function _csr_rowcount_block_gpu(
    rowptr::AbstractVector{Cint}, 
    A::AbstractSparseMatrix{T},
    shift::Int64
) where {T}
    #just add the row count
    kernel = @cuda launch=false _kernel_csr_rowcount_block(rowptr, A, shift)
    config = launch_configuration(kernel.fun)
    threads = min(A.dims[1], config.threads)
    blocks = cld(A.dims[1], threads)

    CUDA.@sync kernel(rowptr, A, shift; threads, blocks)
end

function _kernel_csr_rowcount_block_full(
    rowptr::AbstractVector{Cint}, 
    P::AbstractSparseMatrix{T},
    At::AbstractSparseMatrix{T}
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= P.dims[1]
        rowptr[i+1] += (P.rowPtr[i+1]-P.rowPtr[i]) + (At.rowPtr[i+1]-At.rowPtr[i])
    end

    return nothing
end

@inline function _csr_rowcount_block_full_gpu(
    rowptr::AbstractVector{Cint}, 
    P::AbstractSparseMatrix{T},
    At::AbstractSparseMatrix{T}
) where {T}

    @assert(P.dims[1] == At.dims[1]) #M and N should have the same column number

    #just add the nonzero count each row
    kernel = @cuda launch=false _kernel_csr_rowcount_block_full(rowptr, P, At)
    config = launch_configuration(kernel.fun)
    threads = min(P.dims[1], config.threads)
    blocks = cld(P.dims[1], threads)

    CUDA.@sync kernel(rowptr, P, At; threads, blocks)
end

# #populate a partial column with zeros using the K.colptr as indicator of
# #next fill location in each row.
# function _csr_fill_colvec(K,vtoKKT,initrow,initcol)

#     for i = 1:length(vtoKKT)
#         dest               = K.colptr[initcol]
#         K.rowval[dest]     = initrow + i - 1
#         K.nzval[dest]      = 0.
#         vtoKKT[i]          = dest
#         K.colptr[initcol] += 1
#     end

# end

# #populate a partial row with zeros using the K.colptr as indicator of
# #next fill location in each row.
# function _csr_fill_rowvec(K,vtoKKT,initrow,initcol)

#     for i = 1:length(vtoKKT)
#         col            = initcol + i - 1
#         dest           = K.colptr[col]
#         K.rowval[dest] = initrow
#         K.nzval[dest]  = 0.
#         vtoKKT[i]      = dest
#         K.colptr[col] += 1
#     end

# end


#populate values from M using the rowptr as indicator of
#next fill location in each row.
function _kernel_csr_fill_block(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    M::AbstractSparseMatrix{T}, 
    MtoKKT::AbstractVector{Cint}, 
    initrow::Cint, 
    initcol::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= M.dims[1]
        @inbounds for j = M.rowPtr[i]:(M.rowPtr[i+1]-1)
            row = i + (initrow - 1)
            col = M.colVal[j] + (initcol - 1)

            dest           = rowptr[row]
            colval[dest] = col
            nzval[dest]  = M.nzVal[j]
            MtoKKT[j]      = dest
            rowptr[row] += 1
        end
    end

    return nothing
end

@inline function _csr_fill_block_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    M::AbstractSparseMatrix{T}, 
    MtoKKT::AbstractVector{Cint}, 
    initrow::Cint, 
    initcol::Cint
) where {T}

    kernel = @cuda launch=false _kernel_csr_fill_block(rowptr, colval, nzval, M, MtoKKT, initrow, initcol)
    config = launch_configuration(kernel.fun)
    threads = min(M.dims[1], config.threads)
    blocks = cld(M.dims[1], threads)

    CUDA.@sync kernel(rowptr, colval, nzval, M, MtoKKT, initrow, initcol; threads, blocks)
end

function _csr_fill_specific_diag_full_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    i::Cint
) where {T}
    dest           = rowptr[i]
    colval[dest] = i
    nzval[dest]  = 0        #explicitly fillin 0 values
    rowptr[i] += 1 
end

function _kernel_csr_fill_P_diag_full(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    P::AbstractSparseMatrix{T}, 
    P_zero::AbstractVector{Cint}, 
    PtoKKT::AbstractVector{Cint}
) where {T}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    
    if i <= P.dims[1]
        if(P_zero[i] == -1)    #completely empty column
            dest           = rowptr[i]
            colval[dest] = i
            nzval[dest]  = 0        #explicitly fillin 0 values
            rowptr[i] += 1 
        else
            for j = P.rowPtr[i]:(P.rowPtr[i+1]-1)
                row = i
                col = P.colVal[j]
                if (col == P_zero[i] && col > row)            #insert the diagonal when it is missing in P and P has nonzero entry below the diagonal
                    dest           = rowptr[i]
                    colval[dest] = i
                    nzval[dest]  = 0        #explicitly fillin 0 values
                    rowptr[i] += 1  
                end
                dest           = rowptr[row]
                colval[dest] = col
                nzval[dest]  = P.nzVal[j]
                PtoKKT[j]      = dest
                rowptr[row] += 1
            end
            if (P_zero[i] == 0)            #insert the diagonal when it is missing in P and P has no nonzero entry below the diagonal
                dest           = rowptr[i]
                colval[dest] = i
                nzval[dest]  = 0        #explicitly fillin 0 values
                rowptr[i] += 1 
            end
        end
    end

    return nothing
end

#fill P block with missing diagonal
@inline function _csr_fill_P_diag_full_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    P::AbstractSparseMatrix{T}, 
    P_zero::AbstractVector{Cint}, 
    PtoKKT::AbstractVector{Cint}
) where {T}
    kernel = @cuda launch=false _kernel_csr_fill_P_diag_full(rowptr, colval, nzval, P, P_zero, PtoKKT)
    config = launch_configuration(kernel.fun)
    threads = min(P.dims[1], config.threads)
    blocks = cld(P.dims[1], threads)

    CUDA.@sync kernel(rowptr, colval, nzval, P, P_zero, PtoKKT; threads, blocks)
end

# #Populate the upper or lower triangle with 0s using the K.colptr
# #as indicator of next fill location in each row
# function _csr_fill_dense_triangle(K,blocktoKKT,offset,blockdim,shape)

#     #data will always be supplied as triu, so when filling it into
#     #a tril shape we also need to transpose it.   Just write two
#     #separate cases for clarity here

#     if(shape === :triu)
#         _fill_dense_triangle_triu(K,blocktoKKT,offset,blockdim)
#     else #shape ==== :tril
#         _fill_dense_triangle_tril(K,blocktoKKT,offset,blockdim)
#     end
# end

# function _fill_dense_triangle_triu(K,blocktoKKT,offset,blockdim)

#     kidx = 1
#     for col in offset:(offset + blockdim - 1)
#         for row in (offset:col)
#             dest             = K.colptr[col]
#             K.rowval[dest]   = row
#             K.nzval[dest]    = 0.  #structural zero
#             K.colptr[col]   += 1
#             blocktoKKT[kidx] = dest
#             kidx = kidx + 1
#         end
#     end
# end

# function _fill_dense_triangle_tril(K,blocktoKKT,offset,blockdim)

#     kidx = 1
#     for row in offset:(offset + blockdim - 1)
#         for col in offset:row
#             dest             = K.colptr[col]
#             K.rowval[dest]   = row
#             K.nzval[dest]    = 0.  #structural zero
#             K.colptr[col]   += 1
#             blocktoKKT[kidx] = dest
#             kidx = kidx + 1
#         end
#     end
# end

function _kernel_csr_fill_dense_full(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    Hsblocks::AbstractVector{Cint},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    cone_shift::Cint,
    n::Cint,
    n_rec::Cint
) where {T}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    
    if i <= n_rec
        shift_i = cone_shift+i
        @views block_i = Hsblocks[rng_blocks[shift_i]]
        rng_cone_i = rng_cones[shift_i] .+ n
        idx = 1
        @inbounds for row in rng_cone_i
            @inbounds for col in rng_cone_i
                dest           = rowptr[row]
                colval[dest]   = col
                nzval[dest]    = 0  #structural zero
                rowptr[row]   += 1
                block_i[idx] = dest
                idx = idx + 1
            end
        end
    end

    return nothing
end

@inline function _csr_fill_dense_full_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    Hsblocks::AbstractVector{Cint},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    cone_shift::Cint,
    n::Cint,
    n_rec::Cint
) where {T}
    kernel = @cuda launch=false _kernel_csr_fill_dense_full(rowptr, colval, nzval, Hsblocks, rng_cones, rng_blocks, cone_shift, n, n_rec)
    config = launch_configuration(kernel.fun)
    threads = min(n_rec, config.threads)
    blocks = cld(n_rec, threads)

    CUDA.@sync kernel(rowptr, colval, nzval, Hsblocks, rng_cones, rng_blocks, cone_shift, n, n_rec; threads, blocks)
end

#Populate the diagonal with 0s using the K.colptr as indicator of
#next fill location in each row
function _kernel_csr_fill_diag(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T},
    diagtoKKT::AbstractVector{Cint},
    offset::Cint,
    blockdim::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= blockdim
        row                 = i + offset - 1
        dest                = rowptr[row]
        colval[dest]      = row
        nzval[dest]       = 0.  #structural zero
        rowptr[row]      += 1
        diagtoKKT[i]        = dest
    end

    return nothing
end

@inline function _csr_fill_diag_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T},
    diagtoKKT::AbstractVector{Cint},
    offset::Cint,
    blockdim::Cint
) where {T}

    kernel = @cuda launch=false _kernel_csr_fill_diag(rowptr, colval, nzval, diagtoKKT, offset, blockdim)
    config = launch_configuration(kernel.fun)
    threads = min(blockdim, config.threads)
    blocks = cld(blockdim, threads)

    CUDA.@sync kernel(rowptr, colval, nzval, diagtoKKT, offset, blockdim; threads, blocks)
end

# Populate indices for ut, vt of a sparse second-order cones
function _kernel_csr_fill_sparsecone(
    sparsevec::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    colidx::Cint, 
    rowidx::Cint,
    blockdim::Cint
)

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= blockdim
        sparsevec[i] = rowidx + i
        colval[sparsevec[i]] = colidx + i
    end

    return nothing
end

@inline function _csr_fill_sparsesoc_gpu(
    sparsevec::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    colidx::Cint, 
    rowidx::Cint
)
    blockdim = Cint(length(sparsevec))
    kernel = @cuda launch=false _kernel_csr_fill_sparsecone(sparsevec, colval, colidx, rowidx, blockdim)
    config = launch_configuration(kernel.fun)
    threads = min(blockdim, config.threads)
    blocks = cld(blockdim, threads)

    CUDA.@sync kernel(sparsevec, colval, colidx, rowidx, blockdim; threads, blocks)
end

function _kernel_fill_range!(
    rowptr::AbstractVector{Cint},
    colval::AbstractVector{Cint},
    shift::Cint,
    blockdim::Cint
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= blockdim
        colval[rowptr[i]] = shift + i
    end
    return
end

@inline function _fill_range_gpu(
    rowptr::AbstractVector{Cint},
    colval::AbstractVector{Cint},
    shift::Cint
)   
    blockdim = Cint(length(rowptr))
    kernel = @cuda launch=false _kernel_fill_range!(rowptr, colval, shift, blockdim)
    config = launch_configuration(kernel.fun)
    threads = min(blockdim, config.threads)
    blocks = cld(blockdim, threads)

    CUDA.@sync kernel(rowptr, colval, shift, blockdim; threads, blocks)
end

# #same as _csr_fill_diag, but only places 0.
# #entries where the input matrix M has a missing
# #diagonal entry.  M must be square and TRIU
# function _csr_fill_missing_diag(K,M,initcol)

#     for i = 1:M.n
#         #fill out missing diagonal terms only
#         if((M.colptr[i] == M.colptr[i+1]) ||    #completely empty column
#            (M.rowval[M.colptr[i+1]-1] != i)     #last element is not on diagonal
#           )
#             dest           = K.colptr[i + (initcol - 1)]
#             K.rowval[dest] = i + (initcol - 1)
#             K.nzval[dest]  = 0.  #structural zero
#             K.colptr[i]   += 1
#         end
#     end
# end

function _csr_rowcount_to_rowptr_gpu(
    rowptr::AbstractVector{Cint}
)
    CUDA.@allowscalar rowptr[1] = 1

    #Efficient scan operations for the cumulative sum
    CUDA.accumulate!(+, rowptr, rowptr)
 
end

function _kkt_backshift_colptrs_gpu(rowptr::AbstractVector{Cint})

    rowptr_tmp = deepcopy(rowptr)
    @views copyto!(rowptr[2:end], rowptr_tmp[1:end-1])
    CUDA.@allowscalar rowptr[1] = 1  #zero in C
    rowptr_tmp = nothing                #release the memory

end


function _kernel_map_diag_full(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    diagind::AbstractVector{Cint}
)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    
    if i <= length(diagind)
        @inbounds for j = rowptr[i]:(rowptr[i+1]-1)
            if (colval[j] == i) 
                diagind[i] = j
                return nothing
            end
        end
    end

    return nothing
end

@inline function _map_diag_full_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    diagind::AbstractVector{Cint}
)
    n = length(diagind)
    kernel = @cuda launch=false _kernel_map_diag_full(rowptr, colval, diagind)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(rowptr, colval, diagind; threads, blocks)
end

# function _count_diagonal_entries(P)

#     count = 0
#     for i = 1:P.n

#         #compare last entry in each column with
#         #its row number to identify diagonal entries
#         if((P.colptr[i+1] != P.colptr[i]) &&    #nonempty column
#            (P.rowval[P.colptr[i+1]-1] == i) )   #last element is on diagonal
#                 count += 1
#         end
#     end
#     return count

# end

function _kernel_count_diagonal_entries_full!(P::AbstractSparseMatrix{T}, count::AbstractVector{Bool}) where {T <: AbstractFloat}

	n = size(P,1)

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
	if i <= n
        @inbounds for j = P.rowPtr[i]:P.rowPtr[i+1] -1
            if (P.colVal[j] == i) 
                count[i] = true 
            end
        end
	end
	return nothing
end

function _count_diagonal_entries_full_gpu(P::AbstractSparseMatrix)

    m, n = size(P)
    @assert(m == n)
    count = CuVector{Bool}(undef, m)
    CUDA.fill!(count, false)
    kernel = @cuda launch=false _kernel_count_diagonal_entries_full!(P, count)
    config = launch_configuration(kernel.fun)
    threads = min(m, config.threads)
    blocks = cld(m, threads)

    CUDA.@sync kernel(P, count; threads, blocks)
    total_count = sum(count)
    CUDA.unsafe_free!(count)        #release the memory of unused memory

    return total_count

end