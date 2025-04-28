module DebugArrayAdaptTests

using PartitionedArrays

include(joinpath("..","adapt_tests.jl"))

with_debug(adapt_tests)

end # module
