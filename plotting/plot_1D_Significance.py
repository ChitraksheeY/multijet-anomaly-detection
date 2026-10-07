"""Z_cut vs. number of injected signal events, for CATHODE (solid line, 16-84% band)
and IAD (dashed line), at a background efficiency of 1%.

One plot per signal family (paper: G_KK family and W' -> XY family).
Reads the IADresults.pkl / CATHODEresults.pkl files written by training/train.py,
as produced by training/multijet_SingleSig_injection.sh.

Usage:
  python plot_1D_Significance.py --indir trainings/multijet/SingleSig --family GKK    --output ZNsig_CatIAD_gkk.pdf
  python plot_1D_Significance.py --indir trainings/multijet/SingleSig --family Wprime --output ZNsig_CatIAD_WpXY.pdf
"""
import os
import argparse
import pickle as pkl
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--indir", required=True, help="Directory with one folder per signal (<signal>/<Nsig>/plots/...)")
parser.add_argument("--family", required=True, choices=["GKK", "Wprime"], help="Signal family to plot")
parser.add_argument("--output", required=True, help="Output file (pdf)")
args = parser.parse_args()

beff = 0.01  # background efficiency

allowed_nsig = [0, 600, 1000, 1400, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]

LABEL_MAP = {
    "GKK_qq": r"$G_\mathrm{KK}\to qq$",
    "GKK_gR_Rgg_2217": r"$G_\mathrm{KK}\to gR$,$R\to gg$, $m_R=2217\ \mathrm{GeV}$",
    "GKK_gR_Rgg_500": r"$G_\mathrm{KK}\to gR$, $R\to gg$, $m_R=500\ \mathrm{GeV}$",
    "GKK_gR_Rtt_2217": r"$G_\mathrm{KK}\to gR$, $R\to t\bar t$, $m_R=2217\ \mathrm{GeV}$",
    "GKK_gR_Rtt_500": r"$G_\mathrm{KK}\to gR$, $R\to t\bar t$, $m_R=500\ \mathrm{GeV}$",
    "Wp_XY_X500_Y100": r"$m_X=500, m_Y=100\ \mathrm{GeV}$",
    "Wp_XY_X1000_Y1000": r"$m_X=m_Y=1\ \mathrm{TeV}$",
    "Wp_XY_X1000_Y3000": r"$m_X=1, m_Y=3\ \mathrm{TeV}$",
    "Wp_XY_X2000_Y2000": r"$m_X=m_Y=2\ \mathrm{TeV}$",
}

COLOR_MAP = {
    "GKK_qq": "#1f77b4",
    "GKK_gR_Rgg_2217": "#00a7b9",
    "GKK_gR_Rgg_500": "#2ca02c",
    "GKK_gR_Rtt_500": "#9467bd",
    "GKK_gR_Rtt_2217": "#0b3c5d",
    "Wp_XY_X500_Y100": "#d62728",
    "Wp_XY_X1000_Y1000": "#ff7f0e",
    "Wp_XY_X1000_Y3000": "#e377c2",
    "Wp_XY_X2000_Y2000": "#8c564b",
}

# Signals of each family, in legend order
FAMILIES = {
    "GKK": ["GKK_qq", "GKK_gR_Rgg_2217", "GKK_gR_Rgg_500", "GKK_gR_Rtt_2217", "GKK_gR_Rtt_500"],
    "Wprime": ["Wp_XY_X500_Y100", "Wp_XY_X1000_Y3000", "Wp_XY_X2000_Y2000", "Wp_XY_X1000_Y1000"],
}
signals = FAMILIES[args.family]

patterns = []
for sig in signals:
    for n in allowed_nsig:
        patterns.append(os.path.join(args.indir, sig, f"{n}", "plots", "*_*"))


def parse(pickle):
    with open(pickle, "rb") as file:
        r = pkl.load(file)

    preds = r["preds"].flatten()
    labels = r["labels"]
    nsig = sum(r["Nsig"])
    # signal name from the feature file name, e.g. .../GKK_qq.allfeatures.root -> GKK_qq
    sig = "+".join(os.path.basename(s).split(".")[0] for s in r["sigs"])
    nsig_sr = r["Nsig_SR_train"]
    Nevents = r["events_in_SR"]

    nbkg_sr = Nevents - nsig_sr
    init_Z = nsig_sr / np.sqrt(nbkg_sr) #Z during training

    B = np.sum(labels == 0)
    Z0_val = nsig_sr / np.sqrt(B) # Z0 for changing nsigs, if we had test set with that nsig too

    bkg_preds = preds[labels == 0]
    threshold = np.quantile(bkg_preds, 1 - beff)

    Scut = np.sum((preds > threshold) & (labels != 0))

    seff = Scut / (np.sum(labels != 0) + 1e-10)
    sic = seff / np.sqrt(beff)

    Z = sic * init_Z  #projected Zcut
    return sig, (nsig, Z, Z0_val, sic)


def parse_all(patterns):
    pickles = []
    for pattern in patterns:
        pickles.extend(glob(pattern, recursive=True))

    print(f"Found {len(pickles)} pickles")

    results = {}
    for pickle in tqdm(pickles):
        sig, (nsig, Z, Z0_val, sic) = parse(pickle)
        results.setdefault(sig, {}).setdefault(nsig, []).append((Z, sic, Z0_val))  # nested dict

    return results


def get_curves(nsig_dict, index=0):
    nsigs = sorted(nsig_dict.keys())
    lows, meds, highs = [], [], []
    for ns in nsigs:
        vals = np.array([v[index] for v in nsig_dict[ns]])
        q16, q50, q84 = np.quantile(vals, [0.16, 0.5, 0.84])
        lows.append(q16)
        meds.append(q50)
        highs.append(q84)
    return np.array(nsigs), (np.array(lows), np.array(meds), np.array(highs))


cathode_results = parse_all([f"{p}/CATHODEresults.pkl" for p in patterns])
iad_results = parse_all([f"{p}/IADresults.pkl" for p in patterns])

first_sig = next(iter(iad_results.values()))
nsig_vals = sorted(first_sig.keys())
z0_vals = [first_sig[ns][0][2] for ns in nsig_vals]

for sig in signals:
    if sig not in cathode_results:
        print(f"No CATHODE results for {sig}, skipping")
        continue
    nsig_arr, (low, median, high) = get_curves(cathode_results[sig], index=0)
    label = LABEL_MAP.get(sig, sig)
    color = COLOR_MAP.get(sig, None)
    plt.fill_between(nsig_arr, low, high, alpha=0.2, color=color)
    plt.plot(nsig_arr, median, label=label, color=color, alpha=1.0)

    if sig in iad_results:
        nsig_arr, (low, median, high) = get_curves(iad_results[sig], index=0)
        plt.plot(nsig_arr, median, linestyle='--', color=color, linewidth=2, alpha=1.0)

plt.rcParams.update({
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

ax = plt.gca()

# Interpolate nsig positions where Z0 equals 1, 2, 3, 4, ...
z0_arr = np.array(z0_vals)
nsig_arr = np.array(nsig_vals)
integer_z0s = np.arange(1, int(np.floor(z0_arr.max())) + 1)
nsig_at_integer_z0 = np.interp(integer_z0s, z0_arr, nsig_arr)

secax = ax.secondary_xaxis('top')
secax.set_xlim(ax.get_xlim())
secax.set_xticks(nsig_at_integer_z0)
secax.set_xticklabels([str(int(z)) for z in integer_z0s])
secax.set_xlabel(r"$Z_0$", fontsize=14)

plt.xlabel("Signal injection", fontsize=14)
plt.ylabel(r"$Z_\mathrm{cut}$", fontsize=14)
plt.xlim(0, 3500)
plt.ylim(0, 15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
if args.family == "GKK":
    plt.legend(frameon=False, fontsize=12, labelspacing=0.3, handletextpad=0.3)
else:
    plt.legend(title=r"$W'\to XY\to 4q$", title_fontsize=13, frameon=False, fontsize=13,
               labelspacing=0.3, handletextpad=0.3, loc="lower right")

plt.savefig(args.output, bbox_inches="tight", pad_inches=0.1, dpi=300)
print(f"Saved {args.output}")
