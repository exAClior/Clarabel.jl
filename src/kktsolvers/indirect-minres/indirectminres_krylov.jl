using Krylov

struct MINRESIndirectSolver{T} <: AbstractIndirectMINRESSolver{T}

    solver::MinresQlpSolver{T,T,Vector{T}}
    KKT::SparseMatrixCSC{T,Int}

    #symmetric view for residual calcs
    KKTsym::Symmetric{T, SparseMatrixCSC{T,Int}}

    preconditioner::Vector{T}       #YC: Right now, we use the diagonal preconditioner, 
                                    #    but we need to find a better option 

    # todo: implement settings parsing + delete Dsigns (keeping here just to mirror direct-ldl)
    function MINRESIndirectSolver{T}(KKT0::SparseMatrixCSC{T}, Dsigns, settings) where {T}
        
        dim = LinearAlgebra.checksquare(KKT0) # just to check if square

        solver = MinresQlpSolver(dim,dim,Vector{T})
        preconditioner = ones(dim)

        KKT = deepcopy(KKT0)
        KKTsym = Symmetric(KKT)
        return new(solver,KKT,KKTsym,preconditioner)
    end

end

IndirectMINRESSolversDict[:minres] = MINRESIndirectSolver
required_matrix_shape(::Type{MINRESIndirectSolver}) = :triu

# #update entries in the KKT matrix using the
# #given index into its CSC representation
# function update_values!(
#     minressolver::MINRESIndirectSolver{T},
#     index::AbstractVector{Int},
#     values::Vector{T}
# ) where{T}

#     # no-op. Will just use KKT matrix as it as
#     # passed to refactor!

#     return is_success = true # required to compile for some reason

# end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    minressolver::MINRESIndirectSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}

    # no op. shouldn't need to scale the KKT matrix in this case? 
    _scale_values_KKT!(minressolver.KKT,index,scale)
    
    return is_success = true # required to compile for some reason

end


#refactor the linear system
function refactor!(minressolver::MINRESIndirectSolver{T}, K::SparseMatrixCSC) where{T}

    return is_success = true # required to compile for some reason

end

function update_preconditioner(
    minressolver:: MINRESIndirectSolver{T},
    KKT::Symmetric{T},
) where {T}
    preconditioner = minressolver.preconditioner
    n, m = size(KKT)

    @inbounds for i=1:m
        preconditioner[i] = one(T) / max(abs(KKT[i,i]),1e-8)    # Jacobi preconditioner
    end
end


#solve the linear system
function solve!(
    minressolver:: MINRESIndirectSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    KKT = minressolver.KKTsym

    #solve in place 
    n, m = size(KKT)

    Pinv = Diagonal(minressolver.preconditioner)
    minres_qlp!(minressolver.solver,KKT, b, M= Pinv, Artol=1e-12,atol=1e-12,rtol=1e-13, itmax=10*n)
    x .= minressolver.solver.x

    # todo: note that the second output of minres is the stats, perhaps might be useful to the user

end
