#!/bin/bash

#SBATCH --partition=maxgpu

#SBATCH --time=96:00:00

#SBATCH --array=0-9

#SBATCH --nodes=1

#SBATCH --job-name=multijet_SigMix

#SBATCH --output=logs/multijet_SigMix_%A_%a.out

# Step 2 of the multijet (BB3) study: inject a mixture of two signals, n1 x n2 grid.
# Trains IAD and CATHODE classifiers; CATHODE reuses the flow from multijet_train_cfm.sh.
# 10 runs (array 0-9) per (n1, n2).
# Usage, from the repository root, after multijet_train_cfm.sh has finished:
#   mkdir -p logs && sbatch training/multijet_SigMix_injection.sh

# --- settings (must match multijet_train_cfm.sh) ---
DATADIR=/path/to/feature_files
DATA=$DATADIR/events_LHCO2025_BlackBox3.allfeatures.root
EXTRABKG=$DATADIR/events_LHCO2025_BlackBox3_extrabkg.allfeatures.root
MIN=3000
MAX=4000
MASS=m_rsd
FEATURES=HTtaus
FLOWDIR=trainings/multijet/cfm_${MASS}_${FEATURES}_SR${MIN}to${MAX}

# Signal names as in the paper; feature files are $DATADIR/<name>.allfeatures.root
sig1=GKK_qq               # pythia_BlackBox3_KKg2qq.cmnd
sig2=GKK_gR_Rgg_2217      # pythia_BlackBox3_KKg2gr.cmnd

# Second example pair
#sig1=Wp_XY_X1000_Y1000   # Z_4200_X_1000_Y_1000_qq.cmnd
#sig2=GKK_gR_Rtt_500      # KKg2grtt_r500.cmnd

eval "$(micromamba shell hook --shell bash)"
micromamba activate sk_cathode

for n1 in $(seq 200 200 2200); do
    for n2 in $(seq 200 200 2200); do
        echo "Running ${sig1} x ${sig2}: Nsig = $n1 x $n2 (run $SLURM_ARRAY_TASK_ID)"
        python training/train.py \
            --data $DATA \
            --sigs $DATADIR/$sig1.allfeatures.root $DATADIR/$sig2.allfeatures.root \
            --Nsig $n1 $n2 \
            --extrabkg $EXTRABKG \
            --exBkg 1000000 \
            --min $MIN \
            --max $MAX \
            --mass $MASS \
            --features $FEATURES \
            --outdir trainings/multijet/SigMix/${sig1}_x_${sig2}/${n1}_x_${n2} \
            --flowdir $FLOWDIR \
            --skip-sup \
            --skip-cfm-training \
            --run $SLURM_ARRAY_TASK_ID
    done
done

micromamba deactivate
