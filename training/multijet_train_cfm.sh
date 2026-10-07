#!/bin/bash

#SBATCH --partition=maxgpu

#SBATCH --time=96:00:00

#SBATCH --nodes=1

#SBATCH --job-name=multijet_train_cfm

#SBATCH --output=logs/multijet_train_cfm_%j.out

# Step 1 of the multijet (BB3) study: train the CFM once, without injected signal.
# The trained flow is reused by multijet_SingleSig_injection.sh and the signal mix jobs.
# Usage, from the repository root:
#   mkdir -p logs && sbatch training/multijet_train_cfm.sh

# --- settings ---
DATADIR=/path/to/feature_files
DATA=$DATADIR/events_LHCO2025_BlackBox3.allfeatures.root
EXTRABKG=$DATADIR/events_LHCO2025_BlackBox3_extrabkg.allfeatures.root
MIN=3000
MAX=4000
MASS=m_rsd
FEATURES=HTtaus
FLOWDIR=trainings/multijet/cfm_${MASS}_${FEATURES}_SR${MIN}to${MAX}

eval "$(micromamba shell hook --shell bash)"
micromamba activate sk_cathode

python training/train.py \
    --data $DATA \
    --extrabkg $EXTRABKG \
    --exBkg 1000000 \
    --min $MIN \
    --max $MAX \
    --mass $MASS \
    --features $FEATURES \
    --outdir trainings/multijet/cfm_job \
    --flowdir $FLOWDIR \
    --skip-sup \
    --skip-iad \
    --save-samples \
    --run 0

micromamba deactivate
