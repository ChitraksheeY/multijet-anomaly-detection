#!/bin/bash

#SBATCH --partition=maxgpu

#SBATCH --time=96:00:00

#SBATCH --nodes=1

#SBATCH --job-name=rnd_train_cfm

#SBATCH --output=logs/rnd_train_cfm_%x_%j.out

# Step 1 of the RnD validation: train the CFM once, without injected signal.
# Choose the mass / feature combination with COMBO:
#   mjj_HTtaus   mj1j2, HT + tau1-4,                       SR 3300-3700
#   mjj_Mjtaus   mj1j2, mjmin, delta_mjj, tau21min/max,    SR 3300-3700
#   mrsd_HTtaus  m_rsd, HT + tau1-4,                       SR 2700-3700
#   mrsd_Mjtaus  m_rsd, mjmin, delta_mjj, tau21min/max,    SR 2700-3700
# Usage, from the repository root:
#   mkdir -p logs && sbatch --export=ALL,COMBO=mrsd_HTtaus training/rnd_train_cfm.sh

# --- settings ---
DATADIR=/path/to/feature_files
DATA=$DATADIR/events_LHCO2025_RnD_bkg.allfeatures.root
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

python training/train.py \
    --data $DATA \
    --extrabkg $EXTRABKG \
    --exBkg 1000000 \
    --min $MIN \
    --max $MAX \
    --mass $MASS \
    --features $FEATURES \
    --outdir trainings/rnd/${COMBO}/cfm_job \
    --flowdir $FLOWDIR \
    --skip-sup \
    --skip-iad \
    --run 0

micromamba deactivate
