#!/bin/bash
#SBATCH --time={{time}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --output={{output}}
#SBATCH --error={{error}}

julia --project={{{project}}} -O3 --check-bounds=no -e '
code = quote
   import PartitionedArraysBenchmarks as pb
   import PartitionedArrays as pa
   params = {{{params}}}
   jobname = "{{jobname}}"
   results_dir = "{{{resultsdir}}}"
   pa.with_mpi() do distribute
       pb.experiment(pb.{{{benchmark}}},jobname,distribute,params;results_dir)
   end
end
using MPI
cmd = mpiexec()
run(`$cmd -np {{np}} julia --project={{{project}}} -O3 --check-bounds=no -e $code`)
'
