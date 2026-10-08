# Anomaly detection for multijet scenarios

This repository provides the code and utilities used for studies done for the paper *Anomaly detection for multijet scenarios*, [arXiv:2609.00132](https://arxiv.org/abs/2609.00132).

This wwork applies weakly supervised anomaly detection approach (CATHODE and an Idealised Anomaly Detector, IAD) to multijet final states, based on the signals inspired by LHC Olympics 2020 challenge (specially Black Box 3). The chain is:

1. **Event generation**: All signals and background generation using Pythia 8 + Delphes, converted to LHCO-style HDF5 files.
2. **Feature extraction**: FastJet clustering, Recursive Soft Drop and N-jettiness implementation, and feature calculation, written to ROOT files.
3. **Training**: Supervised, IAD and CATHODE classifiers (CATHODE uses a conditional flow matching model as a ganerative model). All models architectures and usage are defined in the sk_cathode library, utilized here as a submodule for all trainings.
4. **Plotting**: To produce result figures from the paper.

## Repository structure

```
├── data_generation/
│   ├── event_generation/
│   │   ├── cards/                  Pythia cards and the Delphes card
│   │   └── scripts/                produce_qcd.sh, produce_signal.sh, make_dataframe.py, env.sh
│   └── feature_extraction/
│       ├── Makefile
│       ├── src/compute_features.cxx
│       └── scripts/                h5_to_root.py, run_features.sh
├── training/                       train.py and the SLURM job scripts
├── plotting/                       scripts for the paper figures
├── significance_check/             Combine shape fit and counting experiment (Table 1)
├── sk_cathode/                     submodule: the sk_cathode library (multijet branch)
└── environment.yml
```

## Installation

```bash
git clone --recurse-submodules https://github.com/ChitraksheeY/multijet-anomaly-detection.git
cd multijet-anomaly-detection
conda env create -f environment.yml
conda activate multijet_anomaly
pip install -e sk_cathode
```

The data generation needs additional software:

| Step | Software | Version used |
|---|---|---|
| Event generation | Pythia, Delphes | Pythia 8.219, Delphes 3.4.1 (run in an SLC6 container with the LCG_79 view from CVMFS) |
| Feature extraction | ROOT, FastJet, fjcontrib | ROOT 6.40, FastJet 3.4.3, fjcontrib 1.056 (GCC 11.5, RHEL 9) |

FastJet and fjcontrib, for example:

```bash
wget https://fastjet.fr/repo/fastjet-3.4.3.tar.gz && tar xzf fastjet-3.4.3.tar.gz
cd fastjet-3.4.3 && ./configure --prefix=$PWD/../fastjet-install && make -j && make install && cd ..
wget https://fastjet.hepforge.org/contrib/downloads/fjcontrib-1.056.tar.gz && tar xzf fjcontrib-1.056.tar.gz
cd fjcontrib-1.056 && ./configure --fastjet-config=$PWD/../fastjet-install/bin/fastjet-config && make -j && make install
```

`root-config` and `fastjet-config` must be on `PATH` when building the feature extraction.

## 1. Event generation

`data_generation/event_generation/`. Set the Pythia and Delphes paths in `scripts/env.sh`, then from this folder:

```bash
sbatch scripts/produce_qcd.sh 10                   # QCD background
sbatch scripts/produce_signal.sh <card>.cmnd       # one signal sample
```

Each run writes LHCO-style HDF5 files (`samples/h5_files/<process>/run_<id>.h5`): 700 particles × (pT, η, φ) and a label column (0 = background, 1 = signal).

Signals used in the paper:

The sample name is used for the feature files (`<name>.allfeatures.root`), the job scripts and the plot legends.

| Sample name (used in scripts) | Signal process | Pythia card |
|---|---|---|
| `GKK_qq` | $G_\mathrm{KK}\to qq$ | `pythia_BlackBox3_KKg2qq.cmnd` |
| `GKK_gR_Rgg_2217` | $G_\mathrm{KK}\to gR$, $R\to gg$, $m_R=2217$ GeV | `pythia_BlackBox3_KKg2gr.cmnd` |
| `GKK_gR_Rgg_500` | $G_\mathrm{KK}\to gR$, $R\to gg$, $m_R=500$ GeV | `KKg2gr_r500.cmnd` |
| `GKK_gR_Rtt_2217` | $G_\mathrm{KK}\to gR$, $R\to t\bar t$, $m_R=2217$ GeV | `KKg2grtt_r2217.cmnd` |
| `GKK_gR_Rtt_500` | $G_\mathrm{KK}\to gR$, $R\to t\bar t$, $m_R=500$ GeV | `KKg2grtt_r500.cmnd` |
| `Wp_XY_X500_Y100` | $W'\to XY\to 4q$, $m_X=500$, $m_Y=100$ GeV | `Z_4200_X_500_Y_100_qq.cmnd` |
| `Wp_XY_X1000_Y1000` | $W'\to XY\to 4q$, $m_X=m_Y=1$ TeV | `Z_4200_X_1000_Y_1000_qq.cmnd` |
| `Wp_XY_X1000_Y3000` | $W'\to XY\to 4q$, $m_X=1$, $m_Y=3$ TeV | `Z_4200_X_1000_Y_3000_qq.cmnd` |
| `Wp_XY_X2000_Y2000` | $W'\to XY\to 4q$, $m_X=m_Y=2$ TeV | `Z_4200_X_2000_Y_2000_qq.cmnd` |

## 2. Feature extraction

`data_generation/feature_extraction/`:

```bash
make                                                                   # builds compute_features.run
python scripts/h5_to_root.py <h5_dir> <name>.root                     # stitch run_*.h5 into one ROOT file
sbatch scripts/run_features.sh <name>.root <name>.allfeatures.root    # compute the features
```

`compute_features.run` clusters an event-level jet and anti-kT R = 1.0 jets, applies Recursive Soft Drop (RSD), and computes the HT, jettiness features and jet-level features. All output branches are listed in 
`src/compute_features.cxx`. 
The training expects the feature files as `<name>.allfeatures.root`, with the signal names from the table above.

## 3. Training

All job scripts are in `training/` and are submitted from the repository root. Set `DATADIR` in the scripts
to the folder with the feature files. The conditional flow matching model is trained once without signal, as its trained in the sidebands.

| Study | Step 1 | Step 2 |
|---|---|---|
| Multijet, single signals | `multijet_train_cfm.sh` | `multijet_SingleSig_injection.sh` |
| Multijet, signal mixtures | `multijet_train_cfm.sh` | `multijet_SigMix_injection.sh` |
| R&D validation | `rnd_train_cfm.sh` | `rnd_injection.sh` |

```bash
mkdir -p logs
sbatch training/multijet_train_cfm.sh
sbatch training/multijet_SingleSig_injection.sh      
sbatch --export=ALL,COMBO=mrsd_HTtaus training/rnd_train_cfm.sh
sbatch --export=ALL,COMBO=mrsd_HTtaus training/rnd_injection.sh
```

For the R&D study done in the paper, `COMBO` is one of `mjj_HTtaus`, `mjj_Mjtaus`, `mrsd_HTtaus`, `mrsd_Mjtaus`.
`python training/train.py --help` lists all options. Results are saved as `.pkl` files for the plotting scripts.

## 4. Plotting

| Figure | Script |
|---|---|
| 1| `plot_masses.py --set bb3` / `--set other` |
| 4 | `plot_features_SR.py` (needs the generated samples from `multijet_train_cfm.sh`, `--save-samples`) |
| 5 | `plot_1D_Significance.py --family GKK` / `--family Wprime` |
| 6 | `plot_2Dscan_SignalMix.py` |
| 7 | `plot_1D_RnD_Mjj.py` (left), `plot_1D_RnD_Mrsd.py` (right, and the shared legend) |
| Table 1 | `significance_check/shape_analysis.py`, `significance_check/sr_efficiency.py` (see `significance_check/README.md`) |


Other plots in the paper can also be made using the same scripts and modifying the input signal.
Each script has a usage example at the top (`python plotting/<script> --help`).


## Credits

- Pythia and Delphes cards adapted from the LHC Olympics 2020 datasets:
  G. Kasieczka, B. Nachman, D. Shih, *Official Datasets for LHC Olympics 2020 Anomaly Detection Challenge*,
  Zenodo, [doi:10.5281/zenodo.4536624](https://doi.org/10.5281/zenodo.4536624) (CC BY 4.0).
- Combine cards and the new signal cards by Louis Moureaux.
- The [sk_cathode](https://github.com/uhh-pd-ml/sk_cathode) library.
