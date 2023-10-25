include("indirect_diagonal_preconditioner.jl")


# The internal values for the choice of preconditioners
const PreconditionersDict = Dict([
    0           => Clarabel.NoPreconditioner,
    1           => Clarabel.DiagonalPreconditioner,
    2           => Clarabel.NormPreconditioner,
    3           => Clarabel.BlockDiagonalPreconditioner,
])