# Model cards

Pythia 8 cards for the background and signal samples, and the Delphes detector card.
All samples: pp collisions at √s = 13 TeV, `PhaseSpace:pTHatMin = 500` GeV, multiparton interactions off.
The random seed placeholder `NSEED` is replaced by `produce_qcd.sh` / `produce_signal.sh`.

## Background

| Card | Process | Events per job | Source |
|---|---|---|---|
| `pythia_BlackBox3_qcd.cmnd` | QCD 2→2 (`HardQCD:all`, 5 quark flavours), Black Box 3 tune and shower settings | 20000 | LHCO 2020 (Zenodo), seed line added |

## Signals

### $G_\mathrm{KK}$ family

Kaluza–Klein gluon ($G_\mathrm{KK}$, PDG 5100021) with $m = 4200$ GeV, produced via $q\bar q \to G_\mathrm{KK}$.
It decays to a gluon and a radion $R$ (PDG 5100040), or directly to $q\bar q$.

| Name | Card | $G_\mathrm{KK}$ decay | $R$ decay | $m_R$ [GeV] | Events per job | Source |
|---|---|---|---|---|---|---|
| `GKK_qq` | `pythia_BlackBox3_KKg2qq.cmnd` | 100% $d\bar d$ | – | – | 20000 | LHCO 2020 (Zenodo) |
| `GKK_gR_Rgg_2217` | `pythia_BlackBox3_KKg2gr.cmnd` | 99% $gR$, 1% $d\bar d$ | $gg$ | 2217 | 20000 | LHCO 2020 (Zenodo) |
| `GKK_gR_Rgg_500` | `KKg2gr_r500.cmnd` | 99% $gR$, 1% $d\bar d$ | $gg$ | 500 | 10000 | new |
| `GKK_gR_Rtt_2217` | `KKg2grtt_r2217.cmnd` | 99% $gR$, 1% $d\bar d$ | $t\bar t$ | 2217 | 10000 | new |
| `GKK_gR_Rtt_500` | `KKg2grtt_r500.cmnd` | 99% $gR$, 1% $d\bar d$ | $t\bar t$ | 500 | 10000 | new |

In the $R\to t\bar t$ cards the W bosons from the top decays are forced to decay hadronically.

The two Zenodo signal cards have the seed lines commented out, so `produce_signal.sh` does not change their seed.

### $W'$ family

$W'$ (PDG 34) with $m = 4200$ GeV, produced via $f\bar f \to W'$, decaying to $XY$ with X = W (PDG 24) and
Y = Z (PDG 23) with modified masses. Both decay to quarks, giving four-quark final states.

| Name | Card | $m_X$ [GeV] | $m_Y$ [GeV] | Events per job | Source |
|---|---|---|---|---|---|
| `Wp_XY_X500_Y100` | `Z_4200_X_500_Y_100_qq.cmnd` | 500 | 100 | 10000 | new |
| `Wp_XY_X1000_Y1000` | `Z_4200_X_1000_Y_1000_qq.cmnd` | 1000 | 1000 | 10000 | new |
| `Wp_XY_X1000_Y3000` | `Z_4200_X_1000_Y_3000_qq.cmnd` | 1000 | 3000 | 10000 | new |
| `Wp_XY_X2000_Y2000` | `Z_4200_X_2000_Y_2000_qq.cmnd` | 2000 | 2000 | 10000 | new |

## Detector

`delphes_card_BlackBox3.dat`: Delphes card of the LHCO 2020 Black Box 3 (Zenodo), used with Delphes 3.4.1.
Jets are clustered with anti-kT, R = 1.0.

## Source

Cards marked "LHC Olympics 2020" are taken or adapted from G. Kasieczka, B. Nachman, D. Shih,
*Official Datasets for LHC Olympics 2020 Anomaly Detection Challenge*, Zenodo,
[doi:10.5281/zenodo.4536624](https://doi.org/10.5281/zenodo.4536624) (CC BY 4.0).
