#!/usr/bin/env bash
#SBATCH --partition=maxcpu
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --time 1-12:00:00

# Generate QCD background (Pythia8 + Delphes 3.4.1) and convert it to LHCO-style h5.
# Usage, from data_generation/event_generation/:
#   sbatch scripts/produce_qcd.sh N
# Starts one process per CPU core; each runs N times (default 10) with a new seed.
# Output: samples/h5_files/delphes_BlackBox3_qcd/run_<id>.h5

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/env.sh
source /cvmfs/cms.cern.ch/cmsset_default.sh
unset PYTHIA8DATA
mkdir -p samples tmp_cards

function produce() {
    # Unique label for this process
    i=$(echo "$(hostname)$(date +%N)$RANDOM" | md5sum | cut -b 1-6)
    n=${1:-10}
    log_file=samples/qcd_${i}.log
    for j in $(seq $n); do
        cmnd_file=tmp_cards/card_${i}_${j}.cmnd
        root_file=samples/delphes_BlackBox3_qcd_${i}_${j}.root
        sed s/NSEED/${RANDOM}${j}/g <cards/pythia_BlackBox3_qcd.cmnd >$cmnd_file
        echo "Generating events from: $cmnd_file"
        cmssw-slc6 -- sh -c "source /cvmfs/sft.cern.ch/lcg/views/LCG_79/x86_64-slc6-gcc49-opt/setup.sh; cd samples; $DELPHES_DIR/DelphesPythia8 ../cards/delphes_card_BlackBox3.dat ../$cmnd_file ../$root_file"
        (cd samples; python ../scripts/make_dataframe.py --process delphes_BlackBox3_qcd --root_file ../$root_file --index ${i}_${j} --label 0)
        rm $root_file
    done >$log_file
}

for i in $(seq $(nproc)); do
    produce "$@" &
done
wait
