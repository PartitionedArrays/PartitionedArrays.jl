module PartitionedArraysBenchmarkTests

using BenchmarkTools

using SparseArrays
using PartitionedArrays
using LinearAlgebra

function test_benchmark(n)
    """
    Test benchmark with dummy code
    """

    T = SparseMatrixCSC{Float64,Int}
    Ti = indextype(T)
    Tv = eltype(T)
    I = Ti[1,2,5,4,1]
    J = Ti[3,6,1,1,3]
    V = Tv[4,5,3,2,5]
    m = 7
    n = 6
  
    B = compresscoo(T,I,J,V,m,n)

    b1 = ones(Tv,size(B,1))
    b2 = ones(Tv,size(B,1))
    x = collect(Tv,1:size(B,2))
    mul!(b1,B,x)
    spmv!(b2,B,x)
end


# Build a benchmark suite for PartitionedArrays
suite = BenchmarkGroup()
suite["test-suite"] = BenchmarkGroup(["test", "test_tag"])
suite["test-suite"]["n=10"] = @benchmarkable test_benchmark(10)

# Run all benchmarks
tune!(suite)
results = run(suite, verbose = true)

# Save benchmark results for tracking and visualization
BenchmarkTools.save("output.json", median(results))

end # module
