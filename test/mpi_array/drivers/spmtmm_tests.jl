module MPIArrayPrimitivesTests

using PartitionedArrays

include(joinpath("..","..","spmtmm_tests.jl"))

with_mpi(spmtmm_tests)

end # module

