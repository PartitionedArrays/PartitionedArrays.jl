using SparseArrays
using SparseMatricesCSR
using LinearAlgebra
using Printf
using CircularArrays
import MPI
import IterativeSolvers
import Distances

export prefix_sum!
export right_shift!
export length_to_ptrs!
export rewind_ptrs!
export jagged_array
export GenericJaggedArray
export JaggedArray
include("jagged_array.jl")

export nziterator
include("sparse_utils.jl")

export linear_indices
export cartesian_indices
export tuple_of_arrays
export i_am_main
export MAIN
export map_main
export gather
export gather!
export allocate_gather
export scatter
export scatter!
export allocate_scatter
export emit
export emit!
export allocate_emit
export scan
export scan!
export reduction
export reduction!
export ExchangeGraph
export exchange
export exchange_fetch
export exchange!
export exchange_fetch!
export allocate_exchange
include("primitives.jl")

export SequentialData
export with_sequential_data
include("sequential_data.jl")

export MPIData
export mpi_data
export with_mpi_data
include("mpi_data.jl")

export local_range
export boundary_owner
export PRange
export uniform_partition
export variable_partition
export AbstractLocalIndices
export OwnAndGhostIndices
export LocalIndices
export permute_indices
export PermutedLocalIndices
export GhostIndices
export OwnIndices
export local_length
export global_length
export ghost_length
export own_length
export part_id
export local_to_global
export own_to_global
export ghost_to_global
export local_to_owner
export own_to_owner
export ghost_to_owner
export global_to_local
export global_to_own
export global_to_ghost
export own_to_local
export ghost_to_local
export local_to_own
export local_to_ghost
export prange
export replace_ghost
export union_ghost
export find_owner
export Assembler
export vector_assembler
export assemble!
export assembly_buffer_snd
export assembly_buffer_rcv
export to_local!
export to_global!
export partition
export assembly_graph
include("p_range.jl")

export local_values
export own_values
export ghost_values
export allocate_local_values
export OwnAndGhostValues
export PVector
export pvector
export pvector!
export psparsevec!
export pfill
export pzeros
export pones
export prand
export prandn
export consistent!
export neutral_element
include("p_vector.jl")

export PSparseMatrix
export psparse
export psparse!
export own_ghost_values
export ghost_own_values
include("p_sparse_matrix.jl")

export PTimer
export tic!
export toc!
export print_timer
include("p_timer.jl")

