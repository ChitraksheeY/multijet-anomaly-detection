#!/usr/bin/env bash
#SBATCH --partition=maxcpu
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --time 1-12:00:00

# Generate one signal sample (Pythia8 + Delphes 3.4.1) and convert it to LHCO-style h5.
# Usage, from data_generation/event_generation/:
#   sbatch scripts/produce_signal.sh <card.cmnd>     e.g. KKg2grtt_r500.cmnd
# Output: samples/h5_files/delphes_BlackBox3_<card>/run_0.h5

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/env.sh
source /cvmfs/cms.cern.ch/cmsset_default.sh
unset PYTHIA8DATA
mkdir -p samples tmp_cards

card="$1"
log_file=samples/${card/.cmnd/.log}
cmnd_file=tmp_cards/card_${card}
root_file=samples/delphes_BlackBox3_${card/.cmnd/.root}
{
    sed s/NSEED/${RANDOM}/g <cards/$card >$cmnd_file
    echo "Generating events from: $cmnd_file"
    cmssw-slc6 -- sh -c "source /cvmfs/sft.cern.ch/lcg/views/LCG_79/x86_64-slc6-gcc49-opt/setup.sh; cd samples; $DELPHES_DIR/DelphesPythia8 ../cards/delphes_card_BlackBox3.dat ../$cmnd_file ../$root_file"
    (cd samples; python ../scripts/make_dataframe.py --process delphes_BlackBox3_${card/.cmnd/} --root_file ../$root_file --index 0 --label 1)
    rm $root_file
} >$log_file