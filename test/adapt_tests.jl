using Test
using PartitionedArrays
using Adapt

struct FakeCuVector{A} <: AbstractVector{Float64}
    vector::A
end

Base.size(v::FakeCuVector) = size(v.vector)
Base.getindex(v::FakeCuVector,i::Integer) = v.vector[i]

function Adapt.adapt_storage(::Type{<:FakeCuVector},x::AbstractArray)
    FakeCuVector(x)
end

function adapt_tests(distribute)

    rank = distribute(LinearIndices((2,2)))
    
    a = [[1,2],[3,4,5],Int[],[3,4]]
    b = JaggedArray(a)
    c = deepcopy(b)

    c = Adapt.adapt(FakeCuVector,c)

    @test typeof(c.data) == FakeCuVector{typeof(b.data)}
    @test typeof(c.ptrs) == FakeCuVector{typeof(b.ptrs)}
    @test typeof(c).name.wrapper == GenericJaggedArray

    a = [1,2,3,4,5]
    b = deepcopy(a)
    b = Adapt.adapt(FakeCuVector,b)
    @test typeof(b) == FakeCuVector{typeof(a)}
    @test b.vector == a 

    own = [1,2,3,4]
    ghost = [5,6,7,8]
    block_a = split_vector_blocks(own, ghost)
    block_b = deepcopy(block_a)
    block_b = Adapt.adapt(FakeCuVector,block_b)
    @test block_b.own.vector == block_a.own
    @test block_b.ghost.vector == block_a.ghost
    @test typeof(block_b.own) == FakeCuVector{typeof(block_a.own)}
    @test typeof(block_b.ghost) == FakeCuVector{typeof(block_a.ghost)}


    a = split_vector(block_a,[1,2,3,4,5,6,7,8])
    b = deepcopy(a)
    b = Adapt.adapt(FakeCuVector,b)

    @test b.blocks.own.vector == a.blocks.own
    @test b.blocks.ghost.vector == a.blocks.ghost
    @test b.permutation.vector == a.permutation

    
    a = distribute([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    b = distribute([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    b = Adapt.adapt(FakeCuVector,b)

    map(a,b) do val_a,val_b
        @test typeof(val_b) == FakeCuVector{typeof(val_a)}
        @test val_b.vector == val_a
    end
end