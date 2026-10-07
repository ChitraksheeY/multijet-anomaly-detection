#!/bin/bash

#SBATCH --partition=maxgpu

#SBATCH --time=96:00:00

#SBATCH --array=0-9

#SBATCH --nodes=1

#SBATCH --job-name=multijet_SingleSig

#SBATCH --output=logs/multijet_SingleSig_%A_%a.out

# Step 2 of the multijet (BB3) study: inject one signal at a time, with increasing Nsig.
# Trains supervised, IAD and CATHODE classifiers; CATHODE reuses the flow from multijet_train_cfm.sh.
# 10 runs (array 0-9) per signal and Nsig.
# Usage, from the repository root, after multijet_train_cfm.sh has finished:
#   mkdir -p logs && sbatch training/multijet_SingleSig_injection.sh

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
all_sigs=(
    GKK_qq               # pythia_BlackBox3_KKg2qq.cmnd
    GKK_gR_Rgg_2217      # pythia_BlackBox3_KKg2gr.cmnd
    GKK_gR_Rgg_500       # KKg2gr_r500.cmnd
    GKK_gR_Rtt_2217      # KKg2grtt_r2217.cmnd
    GKK_gR_Rtt_500       # KKg2grtt_r500.cmnd
    Wp_XY_X500_Y100      # Z_4200_X_500_Y_100_qq.cmnd
    Wp_XY_X1000_Y1000    # Z_4200_X_1000_Y_1000_qq.cmnd
    Wp_XY_X1000_Y3000    # Z_4200_X_1000_Y_3000_qq.cmnd
    Wp_XY_X2000_Y2000    # Z_4200_X_2000_Y_2000_qq.cmnd
)

eval "$(micromamba shell hook --shell bash)"
micromamba activate sk_cathode

for sig in ${all_sigs[@]}; do
    for n in $(seq 0 200 2000) 2500 3000 3500 4000; do
        echo "Running signal: $sig, Nsig = $n (run $SLURM_ARRAY_TASK_ID)"
        python training/train.py \
            --data $DATA \
            --sigs $DATADIR/$sig.allfeatures.root \
            --Nsig $n \
            --extrabkg $EXTRABKG \
            --exBkg 1000000 \
            --min $MIN \
            --max $MAX \
            --mass $MASS \
            --features $FEATURES \
            --outdir trainings/multijet/SingleSig/$sig/$n \
            --flowdir $FLOWDIR \
            --skip-cfm-training \
            --run $SLURM_ARRAY_TASK_ID
    done
done

micromamba deactivate
