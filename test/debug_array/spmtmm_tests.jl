module DebugArraySpMtMMTests

using PartitionedArrays
using Test

include(joinpath("..","spmtmm_tests.jl"))

v = 1:5
A = sparse(v,v,v)
Z = subtract(A,A)
@test nnz(Z) == nnz(A)
display(Z)

B = sparse(v,v,-v)
Z = add(A,B)
@test nnz(Z) == nnz(A)
display(Z)

A = sparsecsr(v,v,v)
Z = subtract(A,A)
@test nnz(Z) == nnz(A)
display(Z)

B = sparsecsr(v,v,-v)
Z = add(A,B)
@test nnz(Z) == nnz(A)
display(Z)

with_debug(spmtmm_tests)

end # module
