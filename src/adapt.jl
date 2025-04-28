
function Adapt.adapt_structure(to,v::DebugArray)
    v = map(v) do val
         Adapt.adapt_structure(to,val)
    end
end

function Adapt.adapt_structure(to,v::MPIArray)
    v = map(v) do val
         Adapt.adapt_structure(to,val)
    end
end

function Adapt.adapt_structure(to,v::SplitMatrixBlocks)
    own_own = Adapt.adapt(to,v.own_own)
    own_ghost = Adapt.adapt(to,v.own_ghost)
    ghost_ghost = Adapt.adapt(to,v.ghost_ghost)
    ghost_own = Adapt.adapt(to,v.ghost_own)
    split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
end

function Adapt.adapt_structure(to,v::SplitVectorBlocks)
    own = Adapt.adapt(to,v.own)
    ghost = Adapt.adapt(to,v.ghost)
    split_vector_blocks(own,ghost)
end

function Adapt.adapt_structure(to,v::SplitVector)
    blocks = Adapt.adapt(to,v.blocks)
    perm = Adapt.adapt(to,v.permutation)
    split_vector(blocks,perm)
end

function Adapt.adapt_structure(to,v::JaggedArray)
    data = Adapt.adapt_structure(to,v.data)
    ptrs = Adapt.adapt_structure(to,v.ptrs)
    jagged_array(data, ptrs)
end

function Adapt.adapt_structure(to,v::SplitMatrix)
    blocks = Adapt.adapt_structure(to,v.blocks)
    col_per = v.col_permutation
    row_per = v.row_permutation
    split_matrix(blocks,row_per,col_per)
end

function Adapt.adapt_structure(to,v::PSparseMatrix)
    matrix_partition = Adapt.adapt_structure(to,v.matrix_partition)
    col_par = v.col_partition
    row_par = v.row_partition
    PSparseMatrix(matrix_partition,row_par,col_par,v.assembled)
end
