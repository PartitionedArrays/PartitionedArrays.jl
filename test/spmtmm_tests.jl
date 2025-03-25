using SparseArrays
using SparseMatricesCSR
using PartitionedArrays
using LinearAlgebra
using Test

function approx_equivalent(A::SparseMatrixCSC, B::SparseMatrixCSC,args...)
    if size(A) != size(B) && return false; end
    if length(nonzeros(A)) != length(nonzeros(B)) && return false; end
    if A.colptr != B.colptr && return false; end
    if rowvals(A) != rowvals(B) && return false; end
    if !isapprox(nonzeros(A),nonzeros(B),args...) && return false; end
    true
end

# Structurally A and B must be equal, but numerically the can be approximately equal
function approx_equivalent(A::SparseMatrixCSR, B::SparseMatrixCSR,args...)
    if size(A) != size(B) && return false; end
    if length(nonzeros(A)) != length(nonzeros(B)) && return false; end
    if A.rowptr != B.rowptr && return false; end
    if colvals(A) != colvals(B) && return false; end
    if !isapprox(nonzeros(A),nonzeros(B),args...) && return false; end
    true
end

function parallel_tests(pA,pB,sparse_func)
    A = centralize(sparse_func,pA)
    B = centralize(sparse_func,pB)
    # explicit parallel transpose

    pBt = explicit_transpose(pB) |> fetch
    Bt = centralize(sparse_func,pBt)
    @test Bt == copy(transpose(B))
    hp_B = halfperm(B)
    B_struct = symbolic_halfperm(B)
    @test pointer_array(hp_B) == B_struct.ptrs
    @test index_array(hp_B) == B_struct.data
    @test Bt == hp_B

    pBt_local,t = explicit_transpose(pB,reuse=true)
    pBt, transpose_cache = fetch(t)
    Bt = centralize(sparse_func,pBt)
    @test Bt == copy(transpose(B))
    hp_B = halfperm(B)
    @test Bt == hp_B

    t = explicit_transpose!(pBt,pBt_local,pB,transpose_cache)
    wait(t)
    Bt = centralize(sparse_func,pBt)
    @test Bt == copy(transpose(B))
    hp_B = halfperm(B)
    @test Bt == hp_B

    AB0 = matmul(A,B)
    C0 = matmul(transpose(B),AB0)
    # test basic sequential csr implementations to default csc sequential implementations.
    pAB,cacheAB = spmm(pA,pB,reuse=true)
    AB = centralize(sparse_func,pAB)
    @test approx_equivalent(AB,AB0)
    
    # pB will be transposed internally
    pC,cacheC = spmtm(pB,pAB,reuse=true)
    C = centralize(sparse_func,pC)
    @test approx_equivalent(C,C0)
    spmm!(pAB,pA,pB,cacheAB)
    AB = centralize(sparse_func,pAB)

    @test approx_equivalent(AB,AB0)
    spmtm!(pC,pB,pAB,cacheC)
    C = centralize(sparse_func,pC)
    @test approx_equivalent(C,C0)

    pC,cacheC = spmtmm(pB,pA,pB,reuse=true)
    C = centralize(sparse_func,pC)
    @test approx_equivalent(C,C0)

    spmtmm!(pC,pB,pA,pB,cacheC)
    C = centralize(sparse_func,pC)
    @test approx_equivalent(C,C0)
    
    # test basic sequential csr implementations to default csc sequential implementations.
    pC,cacheC = spmm(pBt,pAB,reuse=true)
    C = centralize(sparse_func,pC)
    @test approx_equivalent(C,C0)
    spmm!(pC,pBt,pAB,cacheC)
    C = centralize(sparse_func,pC)
    @test approx_equivalent(C,C0)

    # pB will be transposed internally
    pC,cacheC = spmmm(pBt,pA,pB,reuse=true)
    C = centralize(sparse_func,pC)
    @test approx_equivalent(C,C0)
    spmmm!(pC,pBt,pA,pB,cacheC)
    C = centralize(sparse_func,pC)
    @test approx_equivalent(C,C0)

    # unequal sizes backward (small to large)
    if size(pA) != size(pB)
        CB0 = matmul(C0,Bt)
        D0 = matmul(transpose(Bt),CB0)
        pCB,cacheCB = spmm(pC,pBt,reuse=true)
        CB = centralize(sparse_func,pCB)
        @test approx_equivalent(CB,CB0)

        pD,cacheD = spmtm(pBt,pCB,reuse=true)
        D = centralize(sparse_func,pD)
        @test approx_equivalent(D,D0)
        spmm!(pCB,pC,pBt,cacheCB)
        CB = centralize(sparse_func,pCB)
        @test approx_equivalent(CB,CB0)
        spmtm!(pD,pBt,pCB,cacheD)
        D = centralize(sparse_func,pD)
        @test approx_equivalent(D,D0)
        
        pD,cacheD = spmtmm(pBt,pC,pBt,reuse=true)
        D = centralize(sparse_func,pD)
        @test approx_equivalent(D,D0)
        spmtmm!(pD,pBt,pC,pBt,cacheD)
        D = centralize(sparse_func,pD)
        @test approx_equivalent(D,D0)

        pD,cacheD = spmm(pB,pCB,reuse=true)
        D = centralize(sparse_func,pD)
        @test approx_equivalent(D,D0)
        
        pD,cacheD = spmmm(pB,pC,pBt,reuse=true)
        D = centralize(sparse_func,pD)
        @test approx_equivalent(D,D0)
        spmmm!(pD,pB,pC,pBt,cacheD)
        D = centralize(sparse_func,pD)
        @test approx_equivalent(D,D0)
    end
end

# function parallel_time(pA,pB,sparse_func)
#     A = centralize(sparse_func,pA)
#     B = centralize(sparse_func,pB)
#     # explicit parallel transpose
#     pBt = explicit_transpose(pB) |> fetch
#     Bt = centralize(sparse_func,pBt)
#     @test Bt == copy(transpose(B))
#     hp_B = halfperm(B)
#     @test Bt == hp_B

#     AB0 = A*B
#     C0 = transpose(B)*AB0
#     # test basic sequential csr implementations to default csc sequential implementations.
#     pAB,cacheAB = spmm(pA,pB,reuse=true)
#     print("spmm:\t")
#     @time spmm(pA,pB,reuse=true)
    
#     # pB will be transposed internally
#     pC,cacheC = spmtm(pB,pAB,reuse=true)
#     print("spmtm:\t")
#     @time spmtm(pB,pAB,reuse=true)
#     spmm!(pAB,pA,pB,cacheAB)
#     print("spmm!:\t")
#     @time spmm!(pAB,pA,pB,cacheAB)
#     spmtm!(pC,pB,pAB,cacheC)
#     print("spmtm!:\t")
#     @time spmtm!(pC,pB,pAB,cacheC)
#     # pC,cacheC = spmtmm(pA,pB)
#     pC,cacheC = spmtmm(pB,pA,pB,reuse=true)
#     print("spmtmm:\t")
#     # @time spmtmm(pA,pB)
#     @time spmtmm(pB,pA,pB,reuse=true)
#     # spmtmm!(pC,pA,pB,cacheC)
#     spmtmm!(pC,pB,pA,pB,cacheC)
#     print("spmtmm!:")
#     # @time spmtmm!(pC,pA,pB,cacheC)
#     @time spmtmm!(pC,pB,pA,pB,cacheC)
#     pC,cacheC = spmm(pBt,pAB,reuse=true)
#     print("spmm:\t")
#     @time spmm(pBt,pAB,reuse=true)
#     spmm!(pC,pBt,pAB,cacheC)
#     print("spmm!:\t")
#     @time spmm!(pC,pBt,pAB,cacheC)

#     # pB will be transposed internally
#     pC,cacheC = spmmm(pBt,pA,pB,reuse=true)
#     print("spmmm: ")
#     @time spmmm(pBt,pA,pB,reuse=true)
#     spmmm!(pC,pBt,pA,pB,cacheC)
#     print("spmmm!:")
#     @time spmmm!(pC,pBt,pA,pB,cacheC)

#     # @code_warntype spmmm!(pC,pBt,pA,pB,cacheC)
#     print("Local SpMM:\t")
#     C = A*B
#     @time C = A*B
#     X,cache = rap(Bt,A,B)
#     print("RAP:\t")
#     @time rap(Bt,A,B)
#     rap!(X,Bt,A,B,cache)
#     print("RAP!:\t")
#     @time rap!(X,Bt,A,B,cache)
# end

# function Base.display(A::SparseMatrixCSR)
#     display(halfperm(A) |> PartitionedArrays.ascsc)
# end

function spmtmm_tests(distribute)
    nodes_per_dir = (5,5,5)
    parts_per_dir = (1,2,2)
    np = prod(parts_per_dir)
    ranks = distribute(LinearIndices((np,)))
    for (TiA,TiB,TvA,TvB) in [(Int32,Int32,Float32,Float32),(Int32,Int64,Float32,Float32),(Int32,Int32,Float32,Float64),(Int32,Int64,Float32,Float64),(Int32,Int64,Int64,Int64),(Int32,Int64,Int64,Float32),(Int32,Int64,Float64,Int32)]
        pA = psparse(sparsecsr,laplacian_fdm(nodes_per_dir,parts_per_dir,ranks;index_type=TiA,value_type=TvA)...) |> fetch
        pB = psparse(sparsecsr,laplacian_fdm(nodes_per_dir,parts_per_dir,ranks;index_type=TiB,value_type=TvB)...) |> fetch
        parallel_tests(pA,pB,sparsecsr)

        # Testing with a real prolongator requires PartitionedSolvers
        # T = eltype(typeof(own_own_values(pA).items))
        # pB = prolongator(T,pA)
        # parallel_tests(pA,pB,sparsecsr)
        
        #### CSC ####
        pA = psparse(sparse,laplacian_fdm(nodes_per_dir,parts_per_dir,ranks; index_type = TiA, value_type=TvA)...) |> fetch
        pB = psparse(sparse,laplacian_fdm(nodes_per_dir,parts_per_dir,ranks; index_type = TiB, value_type=TvB)...) |> fetch
        parallel_tests(pA,pB,sparse)

        # Testing with a real prolongator requires PartitionedSolvers
        # T = eltype(typeof(own_own_values(pA).items))
        # pB = prolongator(T,pA)
        # parallel_tests(pA,pB,sparse)
        # break
    end
end