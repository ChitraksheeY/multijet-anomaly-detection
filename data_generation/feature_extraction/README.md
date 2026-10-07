# Feature extraction

`compute_features.run` (built from `src/compute_features.cxx` with `make`) computes jet and event-level
features from the particle-flow candidates of each event (up to 700 particles: pT, η, φ, treated as massless).

## Usage

```bash
make
python scripts/h5_to_root.py <h5_dir> <name>.root                     # stitch run_*.h5 into one ROOT file
sbatch scripts/run_features.sh <name>.root <name>.allfeatures.root    # or run compute_features.run directly
```

Output: a ROOT file with the tree `lhco_blackbox`, one entry per input event, in the same order as the input.

## Settings

| | |
|---|---|
| Jets | anti-kT, R = 1.0 |
| Event jet | all particles clustered into a single jet |
| Grooming | Recursive Soft Drop: z_cut = 0.15, β = 0.7, infinite depth, R0 = 1.0 (dynamical R0) |
| N-subjettiness | one-pass WTA kT axes, unnormalised measure with β = 1|

## Output branches

All masses and momenta in GeV.

| Branch | Description |
|---|---|
| `HT` | scalar sum of the pT of all particles |
| `m_all` | invariant mass of all particles |
| `mj1`, `mj2` | mass of the leading / subleading jet |
| `mj1j2` | invariant mass of the two leading jets ($M_{jj}$) |
| `mj2j3` | invariant mass of the second and third jet |
| `mj1j2j3` | invariant mass of the three leading jets |
| `njet`, `njet_100` | number of jets with pT > 20 / 100 GeV |
| `tau1` … `tau4` | N-subjettiness $\tau_N$ of the event jet |
| `m_rsd` | mass of the groomed event jet ($M_\mathrm{RSD}$) |
| `ht_rsd` | scalar pT sum of the constituents of the groomed event jet |
| `mj1_rsd` … `mj1j2j3_rsd`, `njet_rsd`, `njet_100_rsd`, `tau1_rsd` … `tau4_rsd` | the same quantities computed after grooming |
| `mjmin`, `mjmax` | lighter / heavier of the two leading jet masses |
| `tau21min`, `tau21max` | $\tau_2/\tau_1$ of the lighter / heavier of the two leading jets |
| `delta_mjj` | `mjmax − mjmin` |
| `type` | label copied from the input: 0 = background, 1 = signal |

If an event has fewer than three jets, the missing jets are treated as zero four-vectors.

## Feature sets used in the paper

The first variable is the resonant variable used to define the signal region; the others are the inputs of
the classifiers and the conditional flow (`train.py --mass … --features …`).

| Resonant variable (`--mass`) | Branch |
|---|---|
| $M_\mathrm{RSD}$ | `m_rsd` |
| $M_{jj}$ | `mj1j2` |

| Feature set (`--features`) | Paper | Branches |
|---|---|---|
| `HTtaus` | event-level features | `HT`, `tau1`, `tau2`, `tau3`, `tau4` |
| `Mjtaus` | dijet features | `mjmin`, `delta_mjj`, `tau21min`, `tau21max` |

## Requirements

ROOT (with `root-config`), FastJet 3.4.3 and fjcontrib 1.056 (with `fastjet-config` on `PATH`), a compiler with
OpenMP. Tested with ROOT 6.40 and GCC 11.5 on RHEL 9. The number of threads is set with `OMP_NUM_THREADS`.
