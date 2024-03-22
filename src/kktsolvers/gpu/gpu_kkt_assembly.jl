
#Modified Hsblocks for data type Vector{Vector{T}}
#since CuArray only supports element types that are allocated inline.

abstract type AbstractHsblocksStruct{T <: Real} end
#struct for atomic operations
struct HsblocksGPU{T} <: AbstractHsblocksStruct{T}
    data::AbstractArray{T,1}
    views::AbstractArray{Int,2}

    function HsblocksGPU{T}(cones,data::AbstractArray{T,1}) where {T}
        ncones = length(cones)
        ind = 0
        
        views = CuArray{Int,2}(undef, ncones,2)

        for (i, cone) in enumerate(cones)
            nvars = numel(cone)
            if Hs_is_diagonal(cone) 
                numelblock = nvars
            else #dense block
                numelblock = nvars*nvars #must be Int
            end
            CUDA.@allowscalar views[i,1] = ind + 1      #start index of cone i
            ind += numelblock
            CUDA.@allowscalar views[i,2] = ind          #end index of cone i
        end

        return new(data, views)
    end
end

function _allocate_full_kkt_Hsblocks_gpu(type::Type{T}, cones, Hsblockscpu::ConicHsblocks) where{T <: Real}

    Hsvec = unsafe_wrap(CuArray,Hsblockscpu.vec)
    Hsblocksgpu = HsblocksGPU{T}(cones,Hsvec)

    return Hsblocksgpu
end


# function _assemble_full_kkt_matrix(
#     P::SparseMatrixCSC{T},
#     A::SparseMatrixCSC{T},
#     cones::CompositeCone{T},
#     shape::Symbol = :triu  #or tril
# ) where{T}

#     map   = IndirectDataMap(P,A,cones)
#     (m,n) = (size(A,1), size(P,1))
#     p     = pdim(map.sparse_maps)

#     #entries actually on the diagonal of P
#     nnz_diagP  = _count_diagonal_entries(P)

#     # total entries in the Hs blocks
#     nnz_Hsblocks = length(map.Hsblocks)

#     nnzKKT = (nnz(P) +      # Number of elements in P
#     n -                     # Number of elements in diagonal top left block
#     nnz_diagP +             # remove double count on the diagonal if P has entries
#     2*nnz(A) +                # Number of nonzeros in A and A'
#     nnz_Hsblocks +          # Number of elements in diagonal below A'
#     2*nnz_vec(map.sparse_maps) + # Number of elements in sparse cone off diagonals, 2x compared to the triangle form
#     p                       # Number of elements in diagonal of sparse cones
#     )

#     K = _csc_spalloc(T, m+n+p, m+n+p, nnzKKT)

#     _full_kkt_assemble_colcounts(K,P,A,cones,map)       
#     _full_kkt_assemble_fill(K,P,A,cones,map)

#     return K,map

# end

# function _full_kkt_assemble_colcounts(
#     K::SparseMatrixCSC{T},
#     P::SparseMatrixCSC{T},
#     A::SparseMatrixCSC{T},
#     cones,
#     map
# ) where{T}

#     (m,n) = size(A)

#     #use K.p to hold nnz entries in each
#     #column of the KKT matrix
#     K.colptr .= 0

#     #Count first n columns of KKT
#     _csc_colcount_block_full(K,P,A,1)
#     _csc_colcount_missing_diag_full(K,P,1)
#     _csc_colcount_block(K,A,n+1,:T)

#    # track the next sparse column to fill (assuming triu fill)
#    pcol = m + n + 1 #next sparse column to fill
#    sparse_map_iter = Iterators.Stateful(map.sparse_maps)
    
#     for (i,cone) = enumerate(cones)
#         row = cones.headidx[i] + n

#         #add the the Hs blocks in the lower right
#         blockdim = numel(cone)
#         if Hs_is_diagonal(cone)
#             _csc_colcount_diag(K,row,blockdim)
#         else
#             _csc_colcount_dense_full(K,row,blockdim)
#         end

#         #add sparse expansions columns for sparse cones 
#         if @conedispatch is_sparse_expandable(cone)  
#             thismap = popfirst!(sparse_map_iter)
#             _csc_colcount_sparsecone_full(cone,thismap,K,row,pcol)
#             pcol += pdim(thismap) #next sparse column to fill 
#         end 
#     end

#     return nothing
# end


# function _full_kkt_assemble_fill(
#     K::SparseMatrixCSC{T},
#     P::SparseMatrixCSC{T},
#     A::SparseMatrixCSC{T},
#     cones,
#     map
# ) where{T}

#     (m,n) = size(A)

#     #cumsum total entries to convert to K.p
#     _csc_colcount_to_colptr(K)

#     #filling [P At;A 0] parts
#     _csc_fill_P_block_with_missing_diag_full(K,P,map.P)
#     _csc_fill_block(K,A,map.A,n+1,1,:N)
#     _csc_fill_block(K,A,map.At,1,n+1,:T)

#     # track the next sparse column to fill (assuming full fill)
#     pcol = m + n + 1 #next sparse column to fill
#     sparse_map_iter = Iterators.Stateful(map.sparse_maps)

#     for (i,cone) = enumerate(cones)
#         row = cones.headidx[i] + n

#         #add the the Hs blocks in the lower right
#         blockdim = numel(cone)

#         if Hs_is_diagonal(cone)
#             _csc_fill_diag(K,map.Hsblocks.views[i],row,blockdim)
#         else
#             _csc_fill_dense_full(K,map.Hsblocks.views[i],row,blockdim)
#         end

#         #add sparse expansions columns for sparse cones 
#         if @conedispatch is_sparse_expandable(cone) 
#             thismap = popfirst!(sparse_map_iter)
#             _csc_fill_sparsecone_full(cone,thismap,K,row,pcol)
#             pcol += pdim(thismap) #next sparse column to fill 
#         end 
#     end
    
#     #backshift the colptrs to recover K.p again
#     _kkt_backshift_colptrs(K)

#     #Now we can populate the index of the full diagonal.
#     #We have filled in structural zeros on it everywhere.

#     _map_diag_full(K,map.diag_full)
#     @views map.diagP     .= map.diag_full[1:n]

#     return nothing
# end