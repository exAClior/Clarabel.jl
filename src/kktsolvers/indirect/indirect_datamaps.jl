using SparseArrays, StaticArrays

abstract type SparseExpansionFullMap end 

pdim(maps::Vector{SparseExpansionFullMap}) = sum(pdim, maps; init = 0)
nnz_vec(maps::Vector{SparseExpansionFullMap}) = sum(nnz_vec, maps; init = 0)

struct SOCExpansionFullMap <: SparseExpansionFullMap
    u::Vector{Int}        #off diag dense columns u
    v::Vector{Int}        #off diag dense columns v
    ut::Vector{Int}        #off diag dense columns ut
    vt::Vector{Int}        #off diag dense columns vt
    D::MVector{2, Int}    #diag D
    function SOCExpansionFullMap(cone::SecondOrderCone)
        u = Vector{Int}(undef,numel(cone))
        v = Vector{Int}(undef,numel(cone))
        ut = Vector{Int}(undef,numel(cone))
        vt = Vector{Int}(undef,numel(cone))
        D = MVector(0,0)
        new(u,v,ut,vt,D)
    end
end
pdim(::SOCExpansionFullMap) = 2
nnz_vec(map::SOCExpansionFullMap) = 4*length(map.u)
Dsigns(::SOCExpansionFullMap) = (-1,1)
expansion_fullmap(cone::SecondOrderCone) = SOCExpansionFullMap(cone)

function _csc_colcount_sparsecone_full(
    cone::SecondOrderCone,
    map::SOCExpansionFullMap,
    K::SparseMatrixCSC,
    row::Int,col::Int
)
    
    nvars = numel(cone)

    _csc_colcount_colvec(K,nvars,row, col  ) #v column
    _csc_colcount_colvec(K,nvars,row, col+1) #u column
    _csc_colcount_rowvec(K,nvars,col,   row) #v row
    _csc_colcount_rowvec(K,nvars,col+1, row) #u row

    _csc_colcount_diag(K,col,pdim(map))
end

function _csc_fill_sparsecone_full(
    cone::SecondOrderCone,
    map::SOCExpansionFullMap,
    K::SparseMatrixCSC,row::Int,col::Int
)

    #fill structural zeros for u and v columns for this cone
    #note v is the first extra row/column, u is second
    #fill columns
    _csc_fill_colvec(K, map.v, row, col    ) #v
    _csc_fill_colvec(K, map.u, row, col + 1) #u
    #fill rows
    _csc_fill_rowvec(K, map.vt, col    , row) #vt
    _csc_fill_rowvec(K, map.ut, col + 1, row) #ut

    _csc_fill_diag(K,map.D,col,pdim(map))
end 

function _csc_update_sparsecone_full(
    cone::SecondOrderCone{T},
    map::SOCExpansionFullMap, 
    updateFcn, 
    scaleFcn
) where {T}
    
    η2 = cone.η^2

    #off diagonal columns (or rows)
    updateFcn(map.u,cone.sparse_data.u)
    updateFcn(map.v,cone.sparse_data.v)
    updateFcn(map.ut,cone.sparse_data.u)
    updateFcn(map.vt,cone.sparse_data.v)
    scaleFcn(map.u,-η2)
    scaleFcn(map.v,-η2)
    scaleFcn(map.ut,-η2)
    scaleFcn(map.vt,-η2)

    #set diagonal to η^2*(-1,1) in the extended rows/cols
    updateFcn(map.D,[-η2,+η2])

end

struct GenPowExpansionFullMap <: SparseExpansionFullMap
    
    p::Vector{Int}        #off diag dense columns p
    q::Vector{Int}        #off diag dense columns q
    r::Vector{Int}        #off diag dense columns r
    pt::Vector{Int}        #off diag dense rows pt
    qt::Vector{Int}        #off diag dense rows qt
    rt::Vector{Int}        #off diag dense rows rt
    D::MVector{3, Int}    #diag D

    function GenPowExpansionFullMap(cone::GenPowerCone)
        p = Vector{Int}(undef,numel(cone))
        q = Vector{Int}(undef,dim1(cone))
        r = Vector{Int}(undef,dim2(cone))
        pt = Vector{Int}(undef,numel(cone))
        qt = Vector{Int}(undef,dim1(cone))
        rt = Vector{Int}(undef,dim2(cone))
        D = MVector(0,0,0)
        new(p,q,r,pt,qt,rt,D)
    end
end
pdim(::GenPowExpansionFullMap) = 3
nnz_vec(map::GenPowExpansionFullMap) = (length(map.p) + length(map.q) + length(map.r))<<1
Dsigns(::GenPowExpansionFullMap) = (-1,-1,+1)
expansion_fullmap(cone::GenPowerCone) = GenPowExpansionFullMap(cone)

function _csc_colcount_sparsecone_full(
    cone::GenPowerCone,
    map::GenPowExpansionFullMap,
    K::SparseMatrixCSC,row::Int,col::Int
)

    nvars   = numel(cone)
    dim1 = Clarabel.dim1(cone)
    dim2 = Clarabel.dim2(cone)

    _csc_colcount_colvec(K,dim1, row,        col)    #q column
    _csc_colcount_colvec(K,dim2, row + dim1, col+1)  #r column
    _csc_colcount_colvec(K,nvars,row,        col+2)  #p column

    _csc_colcount_rowvec(K,dim1, col,   row)         #qt row
    _csc_colcount_rowvec(K,dim2, col+1, row + dim1)  #rt row
    _csc_colcount_rowvec(K,nvars,col+2, row)         #pt row

    _csc_colcount_diag(K,col,pdim(map))
    
end

function _csc_fill_sparsecone_full(
    cone::GenPowerCone{T},
    map::GenPowExpansionFullMap,
    K::SparseMatrixCSC{T},
    row::Int,col::Int
) where{T}

    dim1  = Clarabel.dim1(cone)

    _csc_fill_colvec(K, map.q, row,        col)   #q
    _csc_fill_colvec(K, map.r, row + dim1, col+1) #r 
    _csc_fill_colvec(K, map.p, row,        col+2) #p 

    _csc_fill_rowvec(K, map.qt, col,   row)        #qt
    _csc_fill_rowvec(K, map.rt, col+1, row + dim1) #rt
    _csc_fill_rowvec(K, map.pt, col+2, row)        #pt

    _csc_fill_diag(K,map.D,col,pdim(map))

end 

function _csc_update_sparsecone_full(
    cone::GenPowerCone{T},
    map::GenPowExpansionFullMap, 
    updateFcn, 
    scaleFcn
) where {T}
    
    data  = cone.data
    sqrtμ = sqrt(data.μ)

    #off diagonal columns (rows), distribute √μ to off-diagonal terms
    updateFcn(map.q,data.q)
    updateFcn(map.r,data.r)
    updateFcn(map.p,data.p)
    updateFcn(map.qt,data.q)
    updateFcn(map.rt,data.r)
    updateFcn(map.pt,data.p)
    scaleFcn(map.q,-sqrtμ)
    scaleFcn(map.r,-sqrtμ)
    scaleFcn(map.p,-sqrtμ)
    scaleFcn(map.qt,-sqrtμ)
    scaleFcn(map.rt,-sqrtμ)
    scaleFcn(map.pt,-sqrtμ)

    #normalize diagonal terms to 1/-1 in the extended rows/cols
    updateFcn(map.D,[-one(T),-one(T),one(T)])
    
end


struct IndirectDataMap

    P::Vector{Int}
    A::Vector{Int}
    At::Vector{Int}        #YC: not sure whether we need it or not
    Hsblocks::Vector{Vector{Int}}                #indices of the lower RHS blocks (by cone)
    sparse_maps::Vector{SparseExpansionFullMap}      #sparse cone expansion terms

    #all of above terms should be disjoint and their union
    #should cover all of the user data in the KKT matrix.  Now
    #we make two last redundant indices that will tell us where
    #the whole diagonal is, including structural zeros.
    diagP::Vector{Int}
    diag_full::Vector{Int}

    function IndirectDataMap(Pmat::SparseMatrixCSC{T},Amat::SparseMatrixCSC{T},cones) where{T}

        (m,n) = (size(Amat,1), size(Pmat,1))
        P = zeros(Int,nnz(Pmat))
        A = zeros(Int,nnz(Amat))
        At = zeros(Int,nnz(Amat))

        #the diagonal of the ULHS block P.
        #NB : we fill in structural zeros here even if the matrix
        #P is empty (e.g. as in an LP), so we can have entries in
        #index Pdiag that are not present in the index P
        diagP  = zeros(Int,n)

        #make an index for each of the Hs blocks for each cone
        Hsblocks = _allocate_full_kkt_Hsblocks(Int, cones)

        #now do the sparse cone expansion pieces
        nsparse = count(cone->(@conedispatch is_sparse_expandable(cone)),cones)
        sparse_maps = Vector{SparseExpansionFullMap}(); 
        sizehint!(sparse_maps,nsparse)

        for cone in cones
            if @conedispatch is_sparse_expandable(cone) 
                push!(sparse_maps,expansion_fullmap(cone))
            end
        end

        diag_full = zeros(Int,m+n+pdim(sparse_maps))

        return new(P,A,At,Hsblocks,sparse_maps,diagP,diag_full)
    end

end