abstract type AbstractIndirectSolver{T <: AbstractFloat} end

const IndirectSolversDict = Dict{Symbol, UnionAll}()

# Any new indirect solver type should provide implementations of all
# of the following and add itself to the IndirectSolversDict

# register type, .e.g
# IndirectSolversDict[:minres] = MINRESIndirectSolver

# return either :triu or :tril
function required_matrix_shape(::Type{AbstractIndirectSolver})
    error("function not implemented")
end


#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::AbstractIndirectSolver{T},
    index::AbstractVector{Int},
    values::AbstractVector{T}
) where{T}
    error("function not implemented")
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    minressolver::AbstractIndirectSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}
    error("function not implemented")
end


#refactor the linear system
function refactor!(ldlsolver::AbstractIndirectSolver{T}) where{T}
    error("function not implemented")
end


#solve the linear system
function solve!(
    ldlsolver::AbstractIndirectSolver{T},
    x::AbstractVector{T},
    b::AbstractVector{T}
) where{T}
    error("function not implemented")
end


####################################################
# Preconditioner
####################################################
abstract type AbstractPreconditioner{T <: AbstractFloat} <: AbstractMatrix{T} end

function update_preconditioner(
    KKT::SparseMatrixCSC{T,Ti},
    preconditioner::AbstractPreconditioner{T}
) where {T,Ti}
    error("function not implemented")
end