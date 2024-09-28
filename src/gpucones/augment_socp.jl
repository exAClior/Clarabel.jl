
function count_soc(cone, size_soc)

    numel_cone = Clarabel.nvars(cone)
    @assert(numel_cone > size_soc)
        
    num_socs = 1
    numel_cone -= size_soc-1

    while (numel_cone > size_soc-1)
        numel_cone -= size_soc-2
        num_socs += 1
    end

    num_socs += 1

    return num_socs, numel_cone+1
end

#augment At,b for a large soc
function augment_data(At0,
    b0::Vector{T},
    rng_row,size_soc,num_soc,last_size,augx_idx) where {T}

    At = At0[:,rng_row]
    b = b0[rng_row]
    (n,m) = size(At)
    reduce_soc = size_soc - 2
    @assert(reduce_soc > 0)

    bnew = sizehint!(T[],m + 2*(num_soc-1))
    conenew = sizehint!(Clarabel.SupportedCone[],num_soc)

    Atnew = At[:,1]
    push!(bnew,b[1])
    idx = 1             #complete index
    for i in 1:num_soc
        if i == num_soc
            rng = idx+1:idx+last_size-1
            Atnew = hcat(Atnew,At[:,rng])
            @views append!(bnew,b[rng])
            push!(conenew, Clarabel.SecondOrderConeT(last_size))
        else
            rng = idx+1:idx+reduce_soc
            Atnew = hcat(Atnew,At[:,rng])
            @views append!(bnew,b[rng])
            push!(conenew, Clarabel.SecondOrderConeT(size_soc))

            idx += reduce_soc
            augx_idx += 1
            Atnew = hcat(Atnew,sparse([augx_idx, augx_idx],[1,2],[-1,-1],n,2))
            @views append!(bnew,[0,0])
        end
    end

    return Atnew, bnew, conenew, augx_idx
end

function augment_A_b(cones,
    P,
    q::Vector{T},
    A,
    b,
    size_soc,
    num_socs, last_sizes, soc_indices, soc_starts) where {T}

    (m,n) = size(A)
    
    extra_dim = sum(num_socs) - length(num_socs)    #Additional dimensionality for x

    At = vcat(SparseMatrixCSC(A'), spzeros(extra_dim,m))        #May be costly, but more efficient to add rows to a SparseCSR matrix
    bnew = sizehint!(T[],m + 2*extra_dim)
    conesnew = sizehint!(Clarabel.SupportedCone[],length(cones) + extra_dim)

    Atnew = spzeros(n+extra_dim,0)

    start_idx = 0
    end_idx = 0
    cone_idx = 0
    augx_idx = n    #the pointer to the auxiliary x used so far
    
    for (i,ind) in enumerate(soc_indices)

        @views append!(conesnew, cones[cone_idx+1:ind-1])

        numel_cone = Clarabel.nvars(cones[ind])

        end_idx = soc_starts[i]

        rng = start_idx+1:end_idx
        @views Atnew = hcat(Atnew, At[:,rng])
        @views append!(bnew,b[rng])

        start_idx = end_idx
        end_idx += numel_cone
        rng_cone = start_idx+1:end_idx

        Ati,bi,conesi,augx_idx = augment_data(At, b, rng_cone,size_soc,num_socs[i],last_sizes[i],augx_idx)        #augment the current large soc

        Atnew = hcat(Atnew, Ati)
        append!(bnew,bi)
        append!(conesnew,conesi)

        start_idx = end_idx
        cone_idx = ind
    end
    
    if (cone_idx< length(cones))
        @views Atnew = hcat(Atnew, At[:,start_idx+1:end])
        @views append!(bnew,b[start_idx+1:end])
        @views append!(conesnew, cones[cone_idx+1:end])
    end

    Pnew = hcat(vcat(P,spzeros(extra_dim,n)),spzeros(n+extra_dim,extra_dim))
    return Pnew,vcat(q,zeros(extra_dim)),SparseMatrixCSC(Atnew'), bnew, conesnew
end

function expand_soc(cones,size_soc)
    n_large_soc = 0
    soc_indices = sizehint!(Int[],length(cones))            #Indices of large second-order cones 
    soc_starts = sizehint!(Int[],length(cones))          #Starting index of each large second-order cone 
    num_socs = sizehint!(Int[],length(cones))
    last_sizes = sizehint!(Int[],length(cones))      #Size of the last expanded second-order cones 

    cones_dim = 0
    for (i,cone) in enumerate(cones)
        numel_cone = Clarabel.nvars(cone)
        if isa(cone,Clarabel.SecondOrderConeT) && numel_cone > size_soc
            append!(soc_indices, i)
            append!(soc_starts, cones_dim)

            num_soc, last_size = count_soc(cone,size_soc)
            append!(num_socs, num_soc)
            append!(last_sizes, last_size)
            n_large_soc += 1
        end

        cones_dim += numel_cone
    end

    resize!(num_socs,n_large_soc)
    resize!(last_sizes,n_large_soc)

    return num_socs, last_sizes, soc_indices, soc_starts
end
