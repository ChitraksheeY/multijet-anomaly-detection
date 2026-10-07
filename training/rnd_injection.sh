#!/bin/bash

#SBATCH --partition=maxgpu

#SBATCH --time=96:00:00

#SBATCH --array=0-9

#SBATCH --nodes=1

#SBATCH --job-name=rnd_injection

#SBATCH --output=logs/rnd_injection_%A_%a.out

# Step 2 of the RnD validation: inject the RnD signal with increasing Nsig.
# Trains IAD and CATHODE classifiers; CATHODE reuses the flow from rnd_train_cfm.sh (same COMBO).
# 10 runs (array 0-9) per Nsig.
# Usage, from the repository root, after rnd_train_cfm.sh has finished for this COMBO:
#   mkdir -p logs && sbatch --export=ALL,COMBO=mrsd_HTtaus training/rnd_injection.sh

# --- settings (must match rnd_train_cfm.sh) ---
DATADIR=/path/to/feature_files
DATA=$DATADIR/events_LHCO2025_RnD_bkg.allfeatures.root
SIG=$DATADIR/events_LHCO2025_RnD_sig.allfeatures.root
EXTRABKG=$DATADIR/events_LHCO2025_RnD_extrabkg.allfeatures.root

case $COMBO in
    mjj_HTtaus)  MASS=mj1j2; FEATURES=HTtaus; MIN=3300; MAX=3700 ;;
    mjj_Mjtaus)  MASS=mj1j2; FEATURES=Mjtaus; MIN=3300; MAX=3700 ;;
    mrsd_HTtaus) MASS=m_rsd; FEATURES=HTtaus; MIN=2700; MAX=3700 ;;
    mrsd_Mjtaus) MASS=m_rsd; FEATURES=Mjtaus; MIN=2700; MAX=3700 ;;
    *) echo "Set COMBO to one of: mjj_HTtaus, mjj_Mjtaus, mrsd_HTtaus, mrsd_Mjtaus"; exit 1 ;;
esac
FLOWDIR=trainings/rnd/cfm_${MASS}_${FEATURES}_SR${MIN}to${MAX}

eval "$(micromamba shell hook --shell bash)"
micromamba activate sk_cathode

for n in 0 $(seq 500 500 3500); do
    echo "Running RnD ${COMBO}: Nsig = $n (run $SLURM_ARRAY_TASK_ID)"
    python training/train.py \
        --data $DATA \
        --sigs $SIG \
        --Nsig $n \
        --extrabkg $EXTRABKG \
        --exBkg 1000000 \
        --min $MIN \
        --max $MAX \
        --mass $MASS \
        --features $FEATURES \
        --outdir trainings/rnd/${COMBO}/$n \
        --flowdir $FLOWDIR \
        --skip-sup \
        --skip-cfm-training \
        --run $SLURM_ARRAY_TASK_ID
done

micromamba deactivate
