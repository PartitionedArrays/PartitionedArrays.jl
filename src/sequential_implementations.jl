function matmul(A::Union{Transpose{TvA,<:SparseMatrixCSC},<:SparseMatrixCSC} where TvA,
                B::Union{Transpose{TvB,<:SparseMatrixCSC},<:SparseMatrixCSC} where TvB)
    A*B
end

function matmul(A::SparseMatrixCSR,B::SparseMatrixCSR)
    C = matmul(ascsc(B),ascsc(A))
    ascsr(C)
end

function matmul(At::Transpose{Tv,<:SparseMatrixCSR} where Tv,B::SparseMatrixCSR)
    C = matmul(ascsc(B),transpose(ascsc(At.parent)))
    ascsr(C)
end

function matmul(A::SparseMatrixCSR,Bt::Transpose{Tv,<:SparseMatrixCSR} where Tv)
    C = transpose(ascsc(Bt.parent))*ascsc(A)
    ascsr(C)
end

function matmul(At::Transpose{TvA,<:SparseMatrixCSR} where TvA,Bt::Transpose{TvB,<:SparseMatrixCSR} where TvB)
    C = transpose(ascsc(Bt.parent))*transpose(ascsc(At.parent))
    ascsr(C)
end

function mul(x::Number,A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    SparseMatrixCSR{Bi}(size(A)..., copy(A.rowptr), copy(A.colval), map(a -> x*a, A.nzval))
end

function mul(A::SparseMatrixCSR,x::Number) mul(x,A) end

# Alternative to lazy csr to csc for matrix addition that does not drop structural zeros.
function add(A::SparseMatrixCSR{Bi,TvA,TiA},B::SparseMatrixCSR{Bi,TvB,TiB}) where {Bi,TvA,TvB,TiA,TiB}
    if size(A) == size(B) || throw(DimensionMismatch("Size of B $(size(B)) must match size of A $(size(A))"));end
    Ti = promote_type(TiA,TiB)
    Tv = promote_type(TvA,TvB)
    p,q = size(A)
    nnz_C_upperbound = nnz(A) + nnz(B)
    IC = Vector{Ti}(undef, p+1)
    JC = Vector{Ti}(undef, nnz_C_upperbound)
    VC = Vector{Tv}(undef, nnz_C_upperbound)
    
    pC = 1
    JA = colvals(A)
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    for i in 1:p
        IC[i] = pC
        jpA_range = nzrange(A, i)
        jpA, jpA_end = jpA_range.start, jpA_range.stop
        jpB_range = nzrange(B, i)
        jpB, jpB_end = jpB_range.start, jpB_range.stop
        while jpA <= jpA_end && jpB <= jpB_end
            jA = JA[jpA]
            jB = JB[jpB]
            if jA < jB
                JC[pC] = jA
                VC[pC] = VA[jpA]
                jpA += 1
            elseif jB < jA
                JC[pC] = jB
                VC[pC] = VB[jpB]
                jpB += 1
            else
                JC[pC] = jA
                VC[pC] = VA[jpA] + VB[jpB]
                jpA += 1
                jpB += 1
            end
            pC += 1
        end
        while jpA <= jpA_end
            JC[pC] = JA[jpA]
            VC[pC] = VA[jpA]
            jpA += 1
            pC += 1
        end
        while jpB <= jpB_end
            JC[pC] = JB[jpB]
            VC[pC] = VB[jpB]
            jpB += 1
            pC += 1
        end
    end
    IC[end] = pC
    resize!(JC, (pC-1))
    resize!(VC, (pC-1))
    SparseMatrixCSR{Bi}(p,q,IC,JC,VC)   # A += B
end

# Alternative to lazy csr to csc for matrix subtraction that does not drop structural zeros. Subtracts B from A, i.e. A - B.
function subtract(A::SparseMatrixCSR{Bi,TvA,TiA},B::SparseMatrixCSR{Bi,TvB,TiB}) where {Bi,TvA,TvB,TiA,TiB}
    if size(A) == size(B) || throw(DimensionMismatch("Size of B $(size(B)) must match size of A $(size(A))"));end
    Ti = promote_type(TiA,TiB)
    Tv = promote_type(TvA,TvB)
    nnz_C_upperbound = nnz(A) + nnz(B)
    p,r = size(A)
    IC = Vector{Ti}(undef, p+1)
    JC = Vector{Ti}(undef, nnz_C_upperbound)
    VC = Vector{Tv}(undef, nnz_C_upperbound)
    
    pC = 1
    JA = colvals(A)
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    for i in 1:p
        IC[i] = pC
        jpA_range = nzrange(A, i)
        jpA, jpA_end = jpA_range.start, jpA_range.stop
        jpB_range = nzrange(B, i)
        jpB, jpB_end = jpB_range.start, jpB_range.stop
        while jpA <= jpA_end && jpB <= jpB_end
            jA = JA[jpA]
            jB = JB[jpB]
            if jA < jB
                JC[pC] = jA
                VC[pC] = VA[jpA]
                jpA += 1
            elseif jB < jA
                JC[pC] = jB
                VC[pC] = -VB[jpB]
                jpB += 1
            else
                JC[pC] = jA
                VC[pC] = VA[jpA] - VB[jpB]
                jpA += 1
                jpB += 1
            end
            pC += 1
        end
        while jpA <= jpA_end
            JC[pC] = JA[jpA]
            VC[pC] = VA[jpA]
            jpA += 1
            pC += 1
        end
        while jpB <= jpB_end
            JC[pC] = JB[jpB]
            VC[pC] = -VB[jpB]
            jpB += 1
            pC += 1
        end
    end
    IC[end] = pC
    resize!(JC, (pC-1))
    resize!(VC, (pC-1))
    SparseMatrixCSR{Bi}(p,r,IC,JC,VC)   # A += B
end

function subtract(A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    SparseMatrixCSR{Bi}(size(A)..., copy(A.rowptr), copy(A.colval), map(a->-a, A.nzval))
end

# Alternative to lazy csr to csc for matrix addition that does not drop structural zeros.
function add(A::SparseMatrixCSC{TvA,TiA},B::SparseMatrixCSC{TvB,TiB}) where {TvA,TvB,TiA,TiB}
    if size(A) != size(B) && throw(DimensionMismatch("Size of B $(size(B)) must match size of A $(size(A))"));end
    Ti = promote_type(TiA,TiB)
    Tv = promote_type(TvA,TvB)
    p,q = size(A)
    nnz_C_upperbound = nnz(A) + nnz(B)
    JC = Vector{Ti}(undef, q+1)
    IC = Vector{Ti}(undef, nnz_C_upperbound)
    VC = Vector{Tv}(undef, nnz_C_upperbound)
    
    pC = 1
    IA = rowvals(A)
    VA = nonzeros(A)
    IB = rowvals(B)
    VB = nonzeros(B)
    for j in 1:q
        JC[j] = pC
        ipA_range = nzrange(A, j)
        ipA, ipA_end = ipA_range.start, ipA_range.stop
        ipB_range = nzrange(B, j)
        ipB, ipB_end = ipB_range.start, ipB_range.stop
        while ipA <= ipA_end && ipB <= ipB_end
            iA = IA[ipA]
            iB = IB[ipB]
            if iA < iB
                IC[pC] = iA
                VC[pC] = VA[ipA]
                ipA += 1
            elseif iB < iA
                IC[pC] = iB
                VC[pC] = VB[ipB]
                ipB += 1
            else
                IC[pC] = iA
                VC[pC] = VA[ipA] + VB[ipB]
                ipA += 1
                ipB += 1
            end
            pC += 1
        end
        while ipA <= ipA_end
            IC[pC] = IA[ipA]
            VC[pC] = VA[ipA]
            ipA += 1
            pC += 1
        end
        while ipB <= ipB_end
            IC[pC] = IB[ipB]
            VC[pC] = VB[ipB]
            ipB += 1
            pC += 1
        end
    end
    JC[end] = pC
    resize!(IC, (pC-1))
    resize!(VC, (pC-1))
    SparseMatrixCSC{Tv,Ti}(p,q,JC,IC,VC)
end

# Alternative to lazy csr to csc for matrix subtraction that does not drop structural zeros. Subtracts B from A, i.e. A - B.
function subtract(A::SparseMatrixCSC{TvA,TiA},B::SparseMatrixCSC{TvB,TiB}) where {TvA,TvB,TiA,TiB}
    if size(A) == size(B) || throw(DimensionMismatch("Size of B $(size(B)) must match size of A $(size(A))"));end
    Ti = promote_type(TiA,TiB)
    Tv = promote_type(TvA,TvB)
    p,q = size(A)
    nnz_C_upperbound = nnz(A) + nnz(B)
    JC = Vector{Ti}(undef, q+1)
    IC = Vector{Ti}(undef, nnz_C_upperbound)
    VC = Vector{Tv}(undef, nnz_C_upperbound)
    
    pC = 1
    IA = rowvals(A)
    VA = nonzeros(A)
    IB = rowvals(B)
    VB = nonzeros(B)
    for j in 1:q
        JC[j] = pC
        ipA_range = nzrange(A, j)
        ipA, ipA_end = ipA_range.start, ipA_range.stop
        ipB_range = nzrange(B, j)
        ipB, ipB_end = ipB_range.start, ipB_range.stop
        while ipA <= ipA_end && ipB <= ipB_end
            iA = IA[ipA]
            iB = IB[ipB]
            if iA < iB
                IC[pC] = iA
                VC[pC] = VA[ipA]
                ipA += 1
            elseif iB < iA
                IC[pC] = iB
                VC[pC] = VB[ipB]
                ipB += 1
            else
                IC[pC] = iA
                VC[pC] = VA[ipA] - VB[ipB]
                ipA += 1
                ipB += 1
            end
            pC += 1
        end
        while ipA <= ipA_end
            IC[pC] = IA[ipA]
            VC[pC] = VA[ipA]
            ipA += 1
            pC += 1
        end
        while ipB <= ipB_end
            IC[pC] = IB[ipB]
            VC[pC] = -VB[ipB]
            ipB += 1
            pC += 1
        end
    end
    JC[end] = pC
    resize!(IC, (pC-1))
    resize!(VC, (pC-1))
    SparseMatrixCSC{Tv,Ti}(p,q,JC,IC,VC)
end

function subtract(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    SparseMatrixCSC{Tv,Ti}(size(A)..., copy(A.colptr), copy(A.rowval), map(a->-a, A.nzval))
end

function matmul!(C::SparseMatrixCSC,
                 A::SparseMatrixCSC,
                 B::SparseMatrixCSC,
                 cache)
    matmul!(ascsr(C),ascsr(B),ascsr(A),cache)
    C
end

function matmul!(C::SparseMatrixCSC,
                 A::SparseMatrixCSC,
                 B::SparseMatrixCSC,
                 α::Number,
                 β::Number,
                 cache)
    matmul!(ascsr(C),ascsr(B),ascsr(A),α,β,cache)
    C
end

function matmul!(C::SparseMatrixCSC,
                 At::Transpose{Tv,<:SparseMatrixCSC} where Tv,
                 B::SparseMatrixCSC)
    a,b = size(C)
    p,q = size(At)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    A = At.parent
    VC = nonzeros(C)
    VC .= 0
    IC = rowvals(C)
    JA = rowvals(A) # When virtually transposed rowvals represent colvals.
    VA = nonzeros(A)
    IB = rowvals(B)
    VB = nonzeros(B)
    for j in 1:s
        # loop over columns "j" in row i of A
        Bj = nzrange(B, j)
        ptrB_start = Bj.start
        ptrB_stop = Bj.stop
        for ip in nzrange(C, j)
            i = IC[ip]
            # loop over columns "k" in row j of B
            Ai = nzrange(A, i)
            ptrB = ptrB_start
            ptrA = Ai.start
            vC = 0
            while ptrA <= Ai.stop && ptrB <= ptrB_stop
                jA = JA[ptrA]
                iB = IB[ptrB]
                if jA < iB
                    ptrA += 1
                elseif iB < jA
                    ptrB += 1
                else # jA == iB
                    vC += VA[ptrA]*VB[ptrB]
                    ptrA += 1
                    ptrB += 1
                end
            end
            VC[ip] = vC
        end
    end
    C
end

function matmul!(C::SparseMatrixCSC{Tv,Ti},
                 At::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
                 B::SparseMatrixCSC{Tv,Ti},
                 α::Number,
                 β::Number) where {Tv,Ti}
    a,b = size(C)
    p,q = size(At)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    A = At.parent
    VC = nonzeros(C)
    IC = rowvals(C)
    VC .*= β
    JA = rowvals(A) # When virtually transposed rowvals represent colvals.
    VA = nonzeros(A)
    IB = rowvals(B)
    VB = nonzeros(B)
    for j in 1:s
        # loop over columns "j" in row i of A
        Bj = nzrange(B, j)
        for jp in nzrange(C, j)
            i = IC[jp]
            # loop over columns "k" in row j of B
            Ai = nzrange(A, i)
            ptrB = Bj.start
            ptrA = Ai.start
            vC = 0
            while ptrA <= Ai.stop && ptrB <= Bj.stop
                jA = JA[ptrA]
                iB = IB[ptrB]
                if jA == iB
                    vC += VA[ptrA]*VB[ptrB]
                    ptrA += 1
                    ptrB += 1
                elseif jA < iB
                    ptrA += 1
                else
                    ptrB += 1
                end
            end
            VC[jp] += α*vC
        end
    end
    C
end

function matmul!(C::SparseMatrixCSC,
                 A::SparseMatrixCSC,
                 Bt::Transpose{Tv,<:SparseMatrixCSC} where Tv)
    matmul!(ascsr(C),transpose(ascsr(Bt.parent)),ascsr(A))
    C
end

function matmul!(C::SparseMatrixCSR,
                 A::SparseMatrixCSR,
                 B::SparseMatrixCSR,
                 cache)
    a,b = size(C)
    p,q = size(A)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    JC = colvals(C)
    VC = nonzeros(C)
    VC .= zero(eltype(C))
    JA = colvals(A)
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    # A cache here would remove need for allocating acumulating arrays
    # xb = zeros(Ti, p)
    xb,x = cache
    xb .= 0
    # x = similar(xb, Tv) # sparse accumulator, can be zeros() to remove if statement in inner loop.
    for i in 1:p # !
        # loop over rows Ai in col Bj
        for jpa in nzrange(A, i) 
            ja = JA[jpa]
            va = VA[jpa]
            # loop over columns "k" in row j of B
            for jpb in nzrange(B, ja) 
                jb = JB[jpb]
                vb = VB[jpb]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[jb] != i
                    xb[jb] = i
                    x[jb] = va*vb
                else
                    x[jb] += va*vb
                end
            end
        end
        for jpc in nzrange(C,i)
            jc = JC[jpc]
            # To support in-place products whose sparsity patterns are subsets of the sparsity of C, this check is required.
            if xb[jc] == i
                VC[jpc] = x[jc]
            end
        end
    end
    C
end

function matmul!(C::SparseMatrixCSC,
                 A::SparseMatrixCSC,
                 Bt::Transpose{Tv,<:SparseMatrixCSC} where Tv,
                 cache)
    matmul!(ascsr(C),transpose(ascsr(Bt.parent)),ascsr(A),cache)
    C
end

function matmul!(C::SparseMatrixCSR,
                 A::SparseMatrixCSR,
                 B::SparseMatrixCSR,
                 α::Number,
                 β::Number,
                 cache)
    a,b = size(C)
    p,q = size(A)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    JC = colvals(C)
    VC = nonzeros(C)
    VC .*= β
    JA = colvals(A)
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    # A cache here would remove need for allocating acumulating arrays
    # xb = zeros(Ti, p)
    xb,x = cache
    xb .= 0
    # x = similar(xb, Tv) # sparse accumulator, can be zeros() to remove if statement in inner loop.
    for i in 1:p # !
        # loop over rows Ai in col Bj
        for jpa in nzrange(A, i) 
            ja = JA[jpa]
            va = VA[jpa]
            # loop over columns "k" in row j of B
            for jpb in nzrange(B, ja) 
                jb = JB[jpb]
                vb = VB[jpb]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[jb] != i
                    xb[jb] = i
                    x[jb] = va*vb
                else
                    x[jb] += va*vb
                end
            end
        end
        for jpc in nzrange(C,i)
            jc = JC[jpc]
            # To support in-place products whose sparsity patterns are subsets of the sparsity of C, this check is required.
            if xb[jc] == i
                VC[jpc] += α * x[jc]
            end
        end
    end
    C
end

function matmul!(C::SparseMatrixCSC,
                 A::SparseMatrixCSC,
                 Bt::Transpose{Tv,<:SparseMatrixCSC} where Tv,
                 α::Number,
                 β::Number,
                 cache)
    matmul!(ascsr(C),transpose(ascsr(Bt.parent)),ascsr(A),α,β,cache)
    C
end

function matmul!(C::SparseMatrixCSC,
                 At::Transpose{Tv,<:SparseMatrixCSC} where Tv,
                 B::SparseMatrixCSC,
                 cache)
    matmul!(ascsr(C),ascsr(B),transpose(ascsr(At.parent)))
    C
end

function matmul!(C::SparseMatrixCSC,
                 At::Transpose{Tv,<:SparseMatrixCSC} where Tv,
                 B::SparseMatrixCSC,
                 α::Number,
                 β::Number,
                 cache)
    matmul!(ascsr(C),ascsr(A),transpose(ascsr(At.parent)),α,β)
    C
end

# Workaround to supply in-place matmul with auxiliary array, as these are not returned by multiply function exported by SparseArrays
function construct_spmm_cache(A::SparseMatrixCSR{Bi,Tv,Ti} where Bi) where {Tv,Ti}
    q = size(A,2)
    xb = zeros(Ti,q)
    x = similar(xb,Tv)
    xb,x
end
function construct_spmm_cache(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    construct_spmm_cache(ascsr(A))
end

function construct_spmtm_cache(A::SparseMatrixCSR{Bi,Tv,Ti} where Bi) where {Tv,Ti}
    q = size(A,2)
    xb = zeros(Ti,q)
    x = similar(xb,Tv)
    xb,x
end

function construct_spmtm_cache(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    construct_spmtm_cache(ascsr(A))
end

function matmul!(C::SparseMatrixCSR,
                 At::Transpose{Tv,<:SparseMatrixCSR} where Tv,
                 B::SparseMatrixCSR,
                 cache)
    a,b = size(C)
    p,q = size(At)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    A = At.parent
    VC = nonzeros(C)
    VC .= zero((eltype(C)))
    JC = colvals(C)
    JA = colvals(A) # When virtually transposed colvals represent rowvals.
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    xb,x = cache
    xb .= 0
    for k in 1:q
        # loop over columns "j" in row i of B
        for jpb in nzrange(B,k)
            jb = JB[jpb]
            vb = VB[jpb]
            xb[jb] = k
            x[jb] = vb
        end
        for ipa in nzrange(A,k)
            ia = JA[ipa] # interpret column index of A as row index of A^T.
            va = VA[ipa]
            for jpc in nzrange(C, ia)
                jc = JC[jpc]
                # This check is required, as the outerproduct might not contribute to to all nonzero entries in this row of C.
                if xb[jc] == k
                    VC[jpc] += va*x[jc]
                end
            end
        end

    end
    C
end

function matmul!(C::SparseMatrixCSR,
                 At::Transpose{Tv,<:SparseMatrixCSR} where Tv,
                 B::SparseMatrixCSR,
                 α::Number,
                 β::Number,
                 cache)
    a,b = size(C)
    p,q = size(At)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    A = At.parent
    VC = nonzeros(C)
    VC .*= β
    JC = colvals(C)
    JA = colvals(A) # When virtually transposed colvals represent rowvals.
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    xb,x = cache
    xb .= 0
    for k in 1:q
        # loop over columns "j" in row i of B
        for jpb in nzrange(B,k)
            jb = JB[jpb]
            vb = VB[jpb]
            xb[jb] = k
            x[jb] = α*vb
        end
        for ipa in nzrange(A,k)
            ia = JA[ipa] # interpret column index of A as row index of A^T.
            va = VA[ipa]
            for jpc in nzrange(C, ia)
                jc = JC[jpc]
                # This check is required, as the outerproduct might not contribute to to all nonzero entries in this row of C.
                if xb[jc] == k
                    VC[jpc] += va*x[jc]
                end
            end
        end

    end
    C
end

function matmul!(C::SparseMatrixCSR,
                 A::SparseMatrixCSR,
                 Bt::Transpose{Tv,<:SparseMatrixCSR} where Tv)
    matmul!(ascsc(C), transpose(ascsc(Bt.parent)), ascsc(A))
    C
end

function matmul!(C::SparseMatrixCSR,
                 A::SparseMatrixCSR,
                 Bt::Transpose{Tv,<:SparseMatrixCSR} where Tv,
                 α::Number,
                 β::Number)
    matmul!(ascsc(C), transpose(ascsc(Bt.parent)), ascsc(A), α, β)
    C
end

function rap(A::Union{Transpose{TA,<:AbstractSparseMatrix},<:AbstractSparseMatrix} where TA,
             B::M where M<:AbstractSparseMatrix,
             C::Union{Transpose{TC,<:AbstractSparseMatrix},<:AbstractSparseMatrix} where TC
             ;reuse=Val(true))
    D,cache = rap(A,B,C)
    if val_parameter(reuse)
        return D,cache
    end
    D
end

# PtAP variants
function rap(Rt::Transpose{TvR,SparseMatrixCSR{Bi,TvR,TiR}},
             A::SparseMatrixCSR{Bi,TvA,TiA},
             P::SparseMatrixCSR{Bi,TvP,TiP}) where {Bi,TvR,TvA,TvP,TiR,TiA,TiP}
    p,q = size(Rt)
    m,r = size(A)
    n,s = size(P)
    if r != n && throw(DimensionMismatch("Invalid dimensions for A*P: ($m,$r)*($n,$s),"));end
    if q != m && throw(DimensionMismatch("Invalid dimensions: R*AP: ($p,$q)*($m,$s)"));end
    function rap_symbolic_count!(R,A,P)
        Ti = promote_type(TiR,TiA,TiP)
        Tv = promote_type(TvR,TvA,TvP)
        JR = R.data
        JA = colvals(A)
        JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
        xbRA = zeros(Ti, r)
        xbC = zeros(Ti, s) # this vector will also serve as as colptr array in halfperm
        max_rR = find_max_row_length(R)
        max_rA = find_max_row_length(A)
        max_rP = find_max_row_length(P)

        max_rC = max((max_rR*max_rA*max_rP),(max_rA*max_rR))
        JRA = Vector{Ti}(undef,max_rC)
        IC = Vector{Ti}(undef,p+1)
        nnz_C = 1
        IC[1] = nnz_C
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in jagged_range(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            ccC = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(P,j)
                    k = JP[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        ccC += 1
                    end
                end
            end
            nnz_C += ccC
            IC[i+1] = nnz_C
        end
        JC = Vector{Ti}(undef, nnz_C-1)
        VC = zeros(Tv,nnz_C-1)
        JAP = Vector{Ti}(undef,min(max_rA*max_rP,s)) # upper bound estimate for length of virtual row of AP
        xbRA .= 0
        xbC .= 0
        cache = (xbRA,JRA,xbC,JAP)
        SparseMatrixCSR{Bi}(p,s,IC,JC,VC), cache # values not yet initialized
    end
    function rap_symbolic_fill!(C,R,A,P,cache)
        (xbRA,JRA,xbC,JAP) = cache
        JC = colvals(C)
        JR = R.data
        JA = colvals(A)
        JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
        pC = 0
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in jagged_range(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1 
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(P,j)
                    k = JP[kp]
                    if xbC[k] != i
                        pC += 1
                        xbC[k] = i
                        JC[pC] = k
                    end
                end
            end
        end
        xbC .= 0
        outer_cache = (xbC,similar(xbC, eltype(C)),JAP)
        C, outer_cache # values not yet initialized
    end
    function _rap(Rt,A,P)
        R = symbolic_halfperm(Rt.parent)
        C,symbolic_cache = rap_symbolic_count!(R,A,P) # precompute nz structure with a symbolic transpose
        _,outer_cache = rap_symbolic_fill!(C,R,A,P,symbolic_cache)
        Ct = symbolic_halfperm(C)
        symbolic_halfperm!(C,Ct)
        rap!(C,Rt,A,P,outer_cache),(outer_cache...,R)
    end
    _rap(Rt,A,P)
end

function rap(Rt::Transpose{TvR,SparseMatrixCSR{Bi,TvR,TiR}},
             A::SparseMatrixCSR{Bi,TvA,TiA},
             P::SparseMatrixCSR{Bi,TvP,TiP},
             cache) where {Bi,TvR,TvA,TvP,TiR,TiA,TiP}
    p,q = size(Rt)
    m,r = size(A)
    n,s = size(P)
    if r != n && throw(DimensionMismatch("Invalid dimensions for A*P: ($m,$r)*($n,$s),"));end
    if q != m && throw(DimensionMismatch("Invalid dimensions: R*AP: ($p,$q)*($m,$s)"));end

    function rap_symbolic_count(R,A,P)
        Ti = promote_type(TiR,TiA,TiP)
        Tv = promote_type(TvR,TvA,TvP)
        JR = R.data
        JA = colvals(A)
        JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
        xbRA = zeros(Ti, r)
        xbC = zeros(Ti, s) # this vector will also serve as as colptr array in halfperm
        max_rR = find_max_row_length(R)
        max_rA = find_max_row_length(A)
        max_rP = find_max_row_length(P)

        max_rC = max((max_rR*max_rA*max_rP),(max_rA*max_rR))
        JRA = Vector{Ti}(undef,max_rC)
        IC = Vector{Ti}(undef,p+1)
        nnz_C = 1
        IC[1] = nnz_C
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in jagged_range(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            ccC = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(P,j)
                    k = JP[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        ccC += 1
                    end
                end
            end
            nnz_C += ccC
            IC[i+1] = nnz_C
        end
        JC = Vector{Ti}(undef, nnz_C-1)
        VC = zeros(Tv,nnz_C-1)
        JAP = Vector{Ti}(undef,min(max_rA*max_rP,s)) # upper bound estimate for length of virtual row of AP
        xbRA .= 0
        xbC .= 0
        SparseMatrixCSR{Bi}(p,s,IC,JC,VC),(xbRA,JRA,xbC,JAP) # values in CSR matrix not yet initialized
    end
    function rap_symbolic_fill!(C,R,A,P,cache)
        (xbRA,JRA,xbC,JAP) = cache
        JC = colvals(C)
        JR = R.data
        JA = colvals(A)
        JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
        pC = 0
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in jagged_range(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1 
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(P,j)
                    k = JP[kp]
                    if xbC[k] != i
                        pC += 1
                        xbC[k] = i
                        JC[pC] = k
                    end
                end
            end
        end
        xbC .= 0
        C, (xbC,similar(xbC, eltype(C)),JAP) # values not yet initialized
    end
    function _rap(Rt,A,P,old_cache)
        xb,x,JAP,R = old_cache
        old_outer_cache = (xb,x,JAP)
        C,symbolic_cache = rap_symbolic_count(R, A, P)
        _,new_outer_cache = rap_symbolic_fill!(C,R, A, P, symbolic_cache)
        Ct = symbolic_halfperm(C)
        symbolic_halfperm!(C,Ct)
        outer_cache = map((c1,c2) -> length(c1) >= length(c2) ? c1 : c2, old_outer_cache,new_outer_cache)
        rap!(C,Rt,A,P,outer_cache),(outer_cache...,R)
    end
    _rap(Rt,A,P,cache)
end

function reduce_spmtmm_cache(cache,::Type{SparseMatrixCSR})
    (xb,x,JAP,_) = cache
    (xb,x,JAP)
end

function rap!(C::SparseMatrixCSR, 
              Rt::Transpose{Tv,<:SparseMatrixCSR} where Tv,
              A::SparseMatrixCSR,
              P::SparseMatrixCSR,
              cache)
    (a,b) = size(C)
    p,q = size(Rt)
    m,r = size(A)
    n,s = size(P)
    if r != n && throw(DimensionMismatch("Invalid dimensions for A*P: ($m,$r)*($n,$s),"));end
    if q != m && throw(DimensionMismatch("Invalid dimensions: R*AP: ($p,$q)*($m,$s)"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("Dimensions of C $(size(C)) don't match dimensions of R*A*P ($p,$q)*($m,$r)*($n,$s)."));end
    R = Rt.parent
    JC = colvals(C)
    VC = nonzeros(C)
    VC .= zero(eltype(C))

    JA = colvals(A)
    VA = nonzeros(A)
    JP = colvals(P)
    VP = nonzeros(P)
    xb, x, JAP = cache
    xb .= 0
    # loop over rows in A
    for i in 1:m
        lp = 0
        # loop over columns "j" in row i of A
        for jp in nzrange(A, i)
            j = JA[jp]
            va = VA[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(P, j)
                k = JP[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[k] != i
                    lp += 1
                    JAP[lp] = k
                    xb[k] = i
                    x[k] = va * VP[kp]
                else
                    x[k] += va * VP[kp]
                end
            end
        end
        for kp in nzrange(R, i)
            k = colvals(R)[kp] # rowvals when transposed conceptually
            v = nonzeros(R)[kp]
            for jp in nzrange(C,k)
                j = JC[jp]
                if xb[j] == i
                    VC[jp] += v*x[j]
                end 
            end
        end
    end
    C
end

function rap!(C::SparseMatrixCSR,
              Rt::Transpose{Tv,<:SparseMatrixCSR} where Tv,
              A::SparseMatrixCSR,
              P::SparseMatrixCSR,
              α::Number,
              β::Number,
              cache)
    (a,b) = size(C)
    p,q = size(Rt)
    m,r = size(A)
    n,s = size(P)
    if r != n && throw(DimensionMismatch("Invalid dimensions for A*P: ($m,$r)*($n,$s),"));end
    if q != m && throw(DimensionMismatch("Invalid dimensions: R*AP: ($p,$q)*($m,$s)"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("Dimensions of C $(size(C)) don't match dimensions of R*A*P ($p,$q)*($m,$r)*($n,$s)."));end
    R = Rt.parent
    JC = colvals(C)
    VC = nonzeros(C)
    JA = colvals(A)
    VA = nonzeros(A)
    JP = colvals(P)
    VP = nonzeros(P)
    xb, x, JAP = cache
    xb .= 0
    VC .*= β
    # loop over rows in A
    for i in 1:m
        lp = 0
        # loop over columns "j" in row i of A
        for jp in nzrange(A, i)
            j = JA[jp]
            va = α*VA[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(P, j)
                k = JP[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[k] != i
                    lp += 1
                    JAP[lp] = k
                    xb[k] = i
                    x[k] = va*VP[kp]
                else
                    x[k] += va*VP[kp]
                end
            end
        end
        for kp in nzrange(R, i)
            k = colvals(R)[kp] # rowvals when transposed conceptually
            vpl = nonzeros(R)[kp]
            for jp in nzrange(C,k)
                j = JC[jp]
                if xb[j] == i
                    VC[jp] += vpl*x[j]
                end 
            end
        end
    end
    C
end

# RAP variants
function rap(R::SparseMatrixCSR{Bi,TvR,TiR},
             A::SparseMatrixCSR{Bi,TvA,TiA},
             P::SparseMatrixCSR{Bi,TvP,TiP}) where {Bi,TvR,TvA,TvP,TiR,TiA,TiP}
    p,q = size(R)
    m,r = size(A)
    n,s = size(P)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end

    function rap_symbolic!(R,A,P)
        Ti = promote_type(TiR,TiA,TiP)
        Tv = promote_type(TvR,TvA,TvP)

        JR = colvals(R)
        JA = colvals(A)
        JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
        xbRA = zeros(Ti, r)
        xbC = zeros(Ti, s+1) # this vector will also serve as as colptr array in halfperm
        xRA = similar(xbRA, Tv) # sparse accumulator
        xC = similar(xbC, Tv) # sparse accumulator
        max_rR = find_max_row_length(R)
        max_rA = find_max_row_length(A)
        max_rP = find_max_row_length(P)
        max_rC = max((max_rR*max_rA*max_rP),(max_rA*max_rR))

        JRA = Vector{Ti}(undef,max_rC)
        IC = Vector{Ti}(undef,p+1)
        nnz_C = 1
        IC[1] = nnz_C
        for i in 1:p
            ccRA = 0
            # loop over columns "j" in row i of A
            for jp in nzrange(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            ccC = 0
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(P,j)
                    k = JP[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        ccC += 1
                    end
                end
            end
            nnz_C += ccC
            IC[i+1] = nnz_C
        end
        JC = Vector{Ti}(undef, nnz_C-1)
        VC = zeros(Tv,nnz_C-1)
        cache = (xbRA,xRA,JRA,xbC,xC)
        SparseMatrixCSR{Bi}(p,s,IC,JC,VC), cache # values not yet initialized
    end
    function rap_numeric!(C,R,A,P,cache)
        JR = colvals(R)
        VR = nonzeros(R)
        JA = colvals(A)
        VA = nonzeros(A)
        JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
        VP = nonzeros(P)
        JC = colvals(C)
        VC = nonzeros(C)
        (xbRA,xRA,JRA,xbC,xC) = cache
        jpC = 1
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in nzrange(R, i)
                j = JR[jp]
                vpl = VR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                        xRA[k] = vpl * VA[kp]
                    else
                        xRA[k] += vpl * VA[kp]
                    end
                end
            end
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(P,j)
                    k = JP[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        JC[jpC] = k
                        jpC += 1
                        xC[k] = xRA[j]*VP[kp]
                    else
                        xC[k] += xRA[j]*VP[kp]
                    end
                end
            end
            for ind in nzrange(C,i)
                j = JC[ind]
                VC[ind] = xC[j]
            end
        end
    end
    function _rap(R,A,P)
        C,(xbRA,xRA,JRA,xbC,xC) = rap_symbolic!(R,A,P)
        xbRA .= 0
        xbC .= 0
        cache = (xbRA,xRA,JRA,xbC,xC)
        rap_numeric!(C,R,A,P,cache)
        Ct = halfperm!(xbC,similar(colvals(C)),similar(nonzeros(C)),C)
        halfperm!(C,Ct)
        C,cache
    end
    _rap(R,A,P)
end

function reduce_spmtmm_cache(cache,::Type{M} where M <: SparseMatrixCSR)
    (xb,x,JAP,_) = cache
    (xb,x,JAP)
end

function reduce_spmtmm_cache(cache,::Type{M}  where M <: SparseMatrixCSC)
    reduce_spmmmt_cache(cache,SparseMatrixCSR)
end

function rap(R::SparseMatrixCSR{Bi,TvR,TiR},
             A::SparseMatrixCSR{Bi,TvA,TiA},
             P::SparseMatrixCSR{Bi,TvP,TiP},
             cache) where {Bi,TvR,TvA,TvP,TiR,TiA,TiP}
    p,q = size(R)
    m,r = size(A)
    n,s = size(P)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end

    function rap_symbolic!(R,A,P,cache)
        Ti = promote_type(TiR,TiA,TiP)
        Tv = promote_type(TvR,TvA,TvP)
        JR = colvals(R)
        JA = colvals(A)
        JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
        (xbRA,_,JRA,xbC,_) = cache
        IC = Vector{Ti}(undef,p+1)
        nnz_C = 1
        IC[1] = nnz_C
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in nzrange(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            ccC = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(P,j)
                    k = JP[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        ccC += 1
                    end
                end
            end
            nnz_C += ccC
            IC[i+1] = nnz_C
        end
        JC = Vector{Ti}(undef, nnz_C-1)
        VC = zeros(Tv,nnz_C-1)
        SparseMatrixCSR{Bi}(p,s,IC,JC,VC) # values not yet initialized
    end
    function rap_numeric!(C,R,A,P,cache)
        JR = colvals(R)
        VR = nonzeros(R)
        JA = colvals(A)
        VA = nonzeros(A)
        JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
        VP = nonzeros(P)
        JC = colvals(C)
        VC = nonzeros(C)
        (xbRA,xRA,JRA,xbC,xC) = cache
        jpC = 1
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in nzrange(R, i)
                j = JR[jp]
                vpl = VR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                        xRA[k] = vpl * VA[kp]
                    else
                        xRA[k] += vpl * VA[kp]
                    end
                end
            end
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(P,j)
                    k = JP[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        JC[jpC] = k
                        jpC += 1
                        xC[k] = xRA[j]*VP[kp]
                    else
                        xC[k] += xRA[j]*VP[kp]
                    end
                end
            end
            for ind in nzrange(C,i)
                j = JC[ind]
                VC[ind] = xC[j]
            end
        end
    end
    function _rap(R,A,P,old_cache)
        max_rR = find_max_row_length(R)
        max_rA = find_max_row_length(A)
        max_rP = find_max_row_length(P)
        (xbRA,xRA,JRA,xbC,xC) = old_cache
        max_rC = max((max_rR*max_rA*max_rP),(max_rA*max_rR))
        JRA2 = max_rC > length(JRA) ? similar(JRA,max_rC) : JRA
        if r > length(xbRA)
            xbRA2 = similar(xbRA,r)
            xRA2 = similar(xRA,r)
        else
            xbRA2 = xbRA
            xRA2 = xRA
        end

        new_cache = (xbRA2,xRA2,JRA2,xbC,xC)
        xbRA2 .= 0
        xbC .= 0
        C = rap_symbolic!(R,A,P,new_cache)
        xbRA2 .= 0
        xbC .= 0
        rap_numeric!(C,R,A,P,new_cache)
        Ct = halfperm!(xbC,similar(colvals(C)),similar(nonzeros(C)),C)
        halfperm!(C,Ct)
        C,new_cache
    end
    _rap(R,A,P,cache)
end

function reduce_spmmmt_cache(cache,::Type{M} where M <: SparseMatrixCSR)
    (xbRA,xRA,JRA,_,_) = cache
    (xbRA,xRA,JRA)
end

function reduce_spmmmt_cache(cache,::Type{M} where M <: SparseMatrixCSC)
    reduce_spmtmm_cache(cache,SparseMatrixCSR)
end

function rap!(C::SparseMatrixCSR,
              R::SparseMatrixCSR,
              A::SparseMatrixCSR,
              P::SparseMatrixCSR,
              cache)
    p,q = size(R)
    m,r = size(A)
    n,s = size(P)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    JR = colvals(R)
    VR = nonzeros(R)
    JA = colvals(A)
    VA = nonzeros(A)
    JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
    VP = nonzeros(P)
    JC = colvals(C)
    VC = nonzeros(C)
    VC .= zero(eltype(C))
    (xbRA,xRA,JRA,xbC,xC) = cache
    xbRA .= 0
    xbC .= 0
    for i in 1:p
        lp = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
        # loop over columns "j" in row i of A
        for jp in nzrange(R, i)
            j = JR[jp]
            vpl = VR[jp]

            # loop over columns "k" in row j of B
            for kp in nzrange(A, j)
                k = JA[kp]
                va = VA[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xbRA[k] != i
                    lp += 1
                    JRA[lp] = k
                    xbRA[k] = i
                    xRA[k] = vpl * va
                else
                    xRA[k] += vpl * va
                end
            end
        end
        for jp in 1:lp
            j = JRA[jp]
            vra = xRA[j]
            for kp in nzrange(P,j)
                k = JP[kp]
                if xbC[k] != i
                    xbC[k] = i
                    xC[k] = vra*VP[kp]
                else
                    xC[k] += vra*VP[kp]
                end
            end
        end
        for ind in nzrange(C,i)
            j = JC[ind]
            if xbC[j] == i
                VC[ind] = xC[j]
            end
        end
    end
    C
end

function rap!(C::SparseMatrixCSR,
              R::SparseMatrixCSR,
              A::SparseMatrixCSR,
              P::SparseMatrixCSR,
              α::Number,
              β::Number,
              cache)
    p,q = size(R)
    m,r = size(A)
    n,s = size(P)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    JR = colvals(R)
    VR = nonzeros(R)
    JA = colvals(A)
    VA = nonzeros(A)
    JP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
    VP = nonzeros(P)
    JC = colvals(C)
    VC = nonzeros(C)
    VC .*= β
    (xbRA,xRA,JRA,xbC,xC) = cache
    xbRA .= 0
    xbC .= 0
    # xC .= zero(Tv)
    for i in 1:p
        lp = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
        # loop over columns "j" in row i of A
        for jp in nzrange(R, i)
            j = JR[jp]
            vpl = VR[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(A, j)
                k = JA[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xbRA[k] != i
                    lp += 1
                    JRA[lp] = k
                    xbRA[k] = i
                    xRA[k] = vpl * VA[kp]
                else
                    xRA[k] += vpl * VA[kp]
                end
            end
        end
        for jp in 1:lp
            j = JRA[jp]
            for kp in nzrange(P,j)
                k = JP[kp]
                if xbC[k] != i
                    xbC[k] = i
                    xC[k] = xRA[j]*VP[kp]
                else
                    xC[k] += xRA[j]*VP[kp]
                end
            end
        end
        for ind in nzrange(C,i)
            j = JC[ind]
            if xbC[j] == i
                VC[ind] += α*xC[j]
            end
        end
    end
    C
end

# RARt variants
function rap(R::SparseMatrixCSR,
             A::SparseMatrixCSR,
             Pt::Transpose{Tv,<:SparseMatrixCSR} where Tv)
    p,q = size(R)
    m,r = size(A)
    n,s = size(Pt)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions for RA*P: ($p,$r)*($n,$s)"));end
    rap(R,A,copy(Pt))
end

function rap(R::SparseMatrixCSR,
             A::SparseMatrixCSR,
             Pt::Transpose{Tv,<:SparseMatrixCSR} where Tv,
             cache) 
    p,q = size(R)
    m,r = size(A)
    n,s = size(Pt)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions for RA*P: ($p,$r)*($n,$s)"));end
    rap(R,A,copy(Pt),cache)
end

function rap!(C::SparseMatrixCSR,
              R::SparseMatrixCSR, 
              A::SparseMatrixCSR, 
              Pt::Transpose{Tv,<:SparseMatrixCSR} where Tv,
              cache)
    p,q = size(R)
    m,r = size(A)
    n,s = size(Pt)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    P = Pt.parent
    JR = colvals(R)
    VR = nonzeros(R)
    JA = colvals(A)
    VA = nonzeros(A)
    IP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
    VP = nonzeros(P)
    JC = colvals(C)
    VC = nonzeros(C)
    # some cache items are present with the regular rap product in mind, which is how the allocating verison is performed
    xb,x = cache
    xb .= 0
    for i in 1:p
        # loop over columns "j" in row i of A
        for jp in nzrange(R, i)
            j = JR[jp]
            vpl = VR[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(A, j)
                k = JA[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[k] != i
                    xb[k] = i
                    x[k] = vpl * VA[kp]
                else
                    x[k] += vpl * VA[kp]
                end
            end
        end
        for jpP in nzrange(C,i)
            jP = JC[jpP]
            v = zero(eltype(C))
            for ip in nzrange(P,jP)
                iP = IP[ip]
                if xb[iP] == i
                    v += x[iP]*VP[ip]
                end
            end
            VC[jpP] = v
        end
    end
    C
end

function rap!(C::SparseMatrixCSR,
              R::SparseMatrixCSR,
              A::SparseMatrixCSR,
              Pt::Transpose{Tv,<:SparseMatrixCSR} where Tv,
              α::Number,
              β::Number,
              cache)
    p,q = size(R)
    m,r = size(A)
    n,s = size(Pt)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    P = Pt.parent
    JR = colvals(R)
    VR = nonzeros(R)
    JA = colvals(A)
    VA = nonzeros(A)
    IP = colvals(P) # colvals can be interpreted as rowvals when P is virtually transposed.
    VP = nonzeros(P)
    JC = colvals(C)
    VC = nonzeros(C)
    VC .*= β
    # some cache items are present with the regular rap product in mind, which is how the allocating verison is performed
    xb,x = cache
    xb .= 0
    for i in 1:p
        # loop over columns "j" in row i of A
        for jp in nzrange(R, i)
            j = JR[jp]
            vpl = VR[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(A, j)
                k = JA[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[k] != i
                    xb[k] = i
                    x[k] = vpl * VA[kp]
                else
                    x[k] += vpl * VA[kp]
                end
            end
        end
        for jpP in nzrange(C,i)
            jP = JC[jpP]
            v = zero(eltype(C))
            for ip in nzrange(P,jP)
                iP = IP[ip]
                if xb[iP] == i
                    v += x[iP]*VP[ip]
                end
            end
            VC[jpP] += α*v
        end
    end
    C
end

### CSC in terms of CSR
function rap(A::SparseMatrixCSC,
             B::SparseMatrixCSC,
             C::SparseMatrixCSC)
    D,cache = rap(ascsr(C),ascsr(B),ascsr(A))
    ascsc(D),cache
end

function rap(A::SparseMatrixCSC,
             B::SparseMatrixCSC,
             C::SparseMatrixCSC,
             cache)
    D,new_cache = rap(ascsr(C),ascsr(B),ascsr(A),cache)
    ascsc(D),new_cache
end

function rap!(D::SparseMatrixCSC,
              A::SparseMatrixCSC,
              B::SparseMatrixCSC,
              C::SparseMatrixCSC,
              cache)
    rap!(ascsr(D),ascsr(C),ascsr(B),ascsr(A),cache)
    D
end

function rap!(D::SparseMatrixCSC,
              A::SparseMatrixCSC,
              B::SparseMatrixCSC,
              C::SparseMatrixCSC,
              α::Number,
              β::Number,
              cache)
    rap!(ascsr(D),ascsr(C),ascsr(B),ascsr(A),α,β,cache)
    D
end

# PtAP
function rap(A::Transpose{Tv,<:SparseMatrixCSC} where Tv,
             B::SparseMatrixCSC,
             C::SparseMatrixCSC)
    D,cache = rap(ascsr(C),ascsr(B),transpose(ascsr(A.parent)))
    ascsc(D),cache
end

function rap(A::Transpose{Tv,<:SparseMatrixCSC} where Tv,
             B::SparseMatrixCSC,
             C::SparseMatrixCSC,
             cache)
    D,cache = rap(ascsr(C),ascsr(B),transpose(ascsr(A.parent)),cache)
    ascsc(D),cache
end

function rap!(D::SparseMatrixCSC,
              A::Transpose{Tv,<:SparseMatrixCSC} where Tv,
              B::SparseMatrixCSC,
              C::SparseMatrixCSC,
              cache)
    rap!(ascsr(D),ascsr(C),ascsr(B),transpose(ascsr(A.parent)),cache)
    D
end

function rap!(D::SparseMatrixCSC,
              A::Transpose{Tv,<:SparseMatrixCSC} where Tv,
              B::SparseMatrixCSC,
              C::SparseMatrixCSC,
              α::Number,
              β::Number,
              cache)
    rap!(ascsr(D),ascsr(C),ascsr(B),transpose(ascsr(A.parent)),α,β,cache)
    D
end

# RARt
function rap(A::SparseMatrixCSC,
             B::SparseMatrixCSC,
             C::Transpose{Tv,<:SparseMatrixCSC} where Tv)
    D,new_cache = rap(transpose(ascsr(C.parent)),ascsr(B),ascsr(A))
    ascsc(D),new_cache
end
function rap(A::SparseMatrixCSC,
             B::SparseMatrixCSC,
             C::Transpose{Tv,<:SparseMatrixCSC} where Tv,
             cache)
    D,new_cache = rap(transpose(ascsr(C.parent)),ascsr(B),ascsr(A),cache)
    ascsc(D),new_cache
end

function rap!(D::SparseMatrixCSC,
              A::SparseMatrixCSC,
              B::SparseMatrixCSC,
              C::Transpose{Tv,<:SparseMatrixCSC} where Tv,
              cache)
    rap!(ascsr(D),transpose(ascsr(C.parent)),ascsr(B),ascsr(A),cache)
    D
end

function rap!(D::SparseMatrixCSC,
              A::SparseMatrixCSC,
              B::SparseMatrixCSC,
              C::Transpose{Tv,<:SparseMatrixCSC} where Tv,
              α::Number,
              β::Number,
              cache)
    rap!(ascsr(D),transpose(ascsr(C.parent)),ascsr(B),ascsr(A),α,β,cache)
    D
end