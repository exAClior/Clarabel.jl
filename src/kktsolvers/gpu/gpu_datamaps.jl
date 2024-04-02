struct GPUDataMap

    P::AbstractVector{Cint}
    A::AbstractVector{Cint}
    At::AbstractVector{Cint}        #YC: not sure whether we need it or not
    Hsblocks::AbstractVector{Cint}                #indices of the lower RHS blocks (by cone)
    # sparse_maps::Vector{SparseExpansionFullMap}      #YC: disabled sparse cone expansion terms

    #all of above terms should be disjoint and their union
    #should cover all of the user data in the KKT matrix.  Now
    #we make two last redundant indices that will tell us where
    #the whole diagonal is, including structural zeros.
    diagP::AbstractVector{Cint}
    diag_full::AbstractVector{Cint}

    function GPUDataMap(Pmat::SparseMatrixCSC{T},Amat::SparseMatrixCSC{T},cones,mapcpu::IndirectDataMap) where{T}

        (m,n) = (size(Amat,1), size(Pmat,1))
        P = isempty(mapcpu.P) ? CUDA.zeros(Int,0) : unsafe_wrap(CuArray,mapcpu.P)
        A = isempty(mapcpu.A) ? CUDA.zeros(Int,0) : unsafe_wrap(CuArray,mapcpu.A)
        At = isempty(mapcpu.At) ? CUDA.zeros(Int,0) : unsafe_wrap(CuArray,mapcpu.At)

        #the diagonal of the ULHS block P.
        #NB : we fill in structural zeros here even if the matrix
        #P is empty (e.g. as in an LP), so we can have entries in
        #index Pdiag that are not present in the index P
        diagP  = unsafe_wrap(CuArray,mapcpu.diagP)

        #make an index for each of the Hs blocks for each cone
        Hsblocks = CuVector{Cint}(mapcpu.Hsblocks.vec)

        #YC: disable sparse cone expansion at present
        # #now do the sparse cone expansion pieces
        # nsparse = count(cone->(@conedispatch is_sparse_expandable(cone)),cones)
        # sparse_maps = Vector{SparseExpansionFullMap}(); 
        # sizehint!(sparse_maps,nsparse)

        # for cone in cones
        #     if @conedispatch is_sparse_expandable(cone) 
        #         push!(sparse_maps,expansion_fullmap(cone))
        #     end
        # end

        # diag_full = zeros(Int,m+n+pdim(sparse_maps))
        diag_full = unsafe_wrap(CuArray,mapcpu.diag_full)

        return new(P,A,At,Hsblocks,diagP,diag_full)
    end

end