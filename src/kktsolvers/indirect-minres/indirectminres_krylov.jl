import Krylov

struct MINRESIndirectSolver{T} <: AbstractIndirectMINRESSolver{T}

    KKTMatrix::SparseMatrixCSC{T}

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function MINRESIndirectSolver{T}(KKT::SparseMatrixCSC{T}, Dsigns, settings) where {T}
        
        dim = LinearAlgebra.checksquare(KKT) # just to check if square
        # Store the KKT matrix 
        KKTMatrix = KKT
        return new(KKTMatrix)
    end

end

IndirectMINRESSolversDict[:minres] = MINRESIndirectSolver
required_matrix_shape(::Type{MINRESIndirectSolver}) = :Symmetric # note: matrix is already symmetric

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    minressolver::MINRESIndirectSolver{T},
    index::AbstractVector{Int},
    values::Vector{T}
) where{T}

    # no-op. Will just use KKT matrix as it as
    # passed to refactor!

    return is_success = true # required to compile for some reason

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    minressolver::MINRESIndirectSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}

    # no op. shouldn't need to scale the KKT matrix in this case? 
    
    return is_success = true # required to compile for some reason

end


#refactor the linear system
function refactor!(minressolver::MINRESIndirectSolver{T}, K::SparseMatrixCSC) where{T}

    return is_success = true # required to compile for some reason

end


#solve the linear system
function solve!(
    minressolver:: MINRESIndirectSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    #solve in place 
    x .= Krylov.minres(minressolver.KKTMatrix, b)[1]

    # todo: note that the second output of minres is the stats, perhaps might be useful to the user

end
