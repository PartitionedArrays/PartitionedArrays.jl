module MPIArrayAdaptTests

using PartitionedArrays

include(joinpath("..","..","adapt_tests.jl"))

with_mpi(adapt_tests)

end # module
