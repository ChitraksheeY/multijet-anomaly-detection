#!/bin/bash

#SBATCH --partition=maxgpu

#SBATCH --time=60:00:00

#SBATCH --nodes=1

#SBATCH --job-name=compute_features

#SBATCH --output=%x.txt  

#SBATCH --error=%x.txt  

#SBATCH --mail-type=END 

# Usage, from data_generation/feature_extraction/ (build first with: make):
#   sbatch scripts/run_features.sh <input.root> <output.root>
# Number of threads: all cores of the node, or set OMP_NUM_THREADS.

unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge

./compute_features.run "$1" "$2" lhco_blackbox

echo "compute_features.run is executed!"
