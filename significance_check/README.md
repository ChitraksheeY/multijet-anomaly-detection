# Significance check
(The scripts)

This scripts let you perform intial significances without any ML based anomaly detection: a simple counting experiment in the signal region and a
binned shape fit of the mass spectrum with [Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/). This is described in details in the paper, section 3.2.
Both use the feature files (`<name>.allfeatures.root`) and scale background and signal to the expected
yields (default: 996 800 background and 3 200 signal events, as in Black Box 3 dataset).

| Script | What it does |
|---|---|
| `sr_efficiency.py` | counts background and signal in a mass window, prints the efficiencies and $S/\sqrt{B}$ |
| `shape_analysis.py` | histograms the mass (default: 500 bins), writes a Combine datacard and shapes, runs `combine -M Significance` |

Both accept a second signal (`--signal2`, `--nsig2`) for signal mixtures.

## Settings used to reproduce Table 1

| Mass | `shape_analysis.py` | `sr_efficiency.py` |
|---|---|---|
| $M_\mathrm{RSD}$ | default (`--branch m_rsd --min 2000 --max 7000`) | default (`--branch m_rsd --min 3000 --max 4000`) |
| $M_{jj}$ | `--branch mj1j2 --min 2500 --max 7000` | `--branch mj1j2 --min 4000 --max 4400` |
| $M_\mathrm{all}$ | `--branch m_all --min 3000 --max 9000` | `--branch m_all --min 4000 --max 5000` |

## Usage

From the `significance_check/` folder:

```bash
# counting experiment, M_RSD
python sr_efficiency.py --background <background>.allfeatures.root --signal GKK_qq.allfeatures.root

# shape fit, M_jj (needs Combine, see below)
source env.sh
python shape_analysis.py --background <background>.allfeatures.root --signal GKK_qq.allfeatures.root \
    --branch mj1j2 --min 2500 --max 7000

# signal mixture, e.g. as in Black Box 3
python shape_analysis.py --background <background>.allfeatures.root \
    --signal GKK_qq.allfeatures.root --nsig 1200 --signal2 GKK_gR_Rgg_2217.allfeatures.root --nsig2 2000
```

The datacard contains two log-normal uncertainties: 10% on the signal yield (`lumi`) and 30% on the background
normalisation (`bgnorm`). `example_cards/` has the generated cards and shapes for $M_\mathrm{RSD}$, $M_{jj}$ and
$M_\mathrm{all}$, for a $G_\mathrm{KK}$ signal (`card*.txt`) and a $W'$ signal (`card_W_*.txt`).

## Combine setup

The results were produced with Combine in `CMSSW_14_1_0_pre4` (see the
[Combine documentation](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/) for installation).
`env.sh` sets up the environment from CVMFS; source it from the folder that contains `CMSSW_14_1_0_pre4/`.

