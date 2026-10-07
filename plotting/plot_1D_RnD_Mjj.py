"""R&D validation with M_jj as resonant variable (paper: Figure 7, left).

Z_cut vs. number of injected signal events for CATHODE (solid line, 16-84% band)
and IAD (dashed line), at a background efficiency of 1%, for M_jj + event-level
features and M_jj + dijet features. The legend is a separate panel, made by plot_1D_RnD_Mrsd.py.

Reads the results of training/rnd_injection.sh (COMBO=mjj_HTtaus and COMBO=mjj_Mjtaus).

Usage:
  python plot_1D_RnD_Mjj.py --indir trainings/rnd --output RnDwithMjj.pdf
"""
import os
import argparse
import pickle as pkl
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--indir", required=True, help="Directory with one folder per COMBO (<COMBO>/<Nsig>/plots/...)")
parser.add_argument("--output", required=True, help="Output file (pdf)")
args = parser.parse_args()

beff = 0.01  # background efficiency

LABEL_MAP = {
    "mjj_HTtaus": r"$M_{jj}$ + event-level features",
    "mjj_Mjtaus": r"$M_{jj}$ + dijet features",
}

COLOR_MAP = {
    "mjj_HTtaus": "#7B2D8B",
    "mjj_Mjtaus": "#117A7A",
}


def parse_all(pickles):
    print(f"Found {len(pickles)} pickles")

    results = {}

    for pickle in tqdm(pickles):
        with open(pickle, "rb") as file:
            r = pkl.load(file)

        sig = pickle.split(os.sep)[-5]  # <indir>/<COMBO>/<Nsig>/plots/<name>/<file>.pkl

        preds = r["preds"].flatten()
        labels = r["labels"]
        nsig = sum(r["Nsig"])           #injected during training
        nsig_sr = r["Nsig_SR_train"]    #during training
        Nevents = r["events_in_SR"]     #also during training

        nbkg_sr = Nevents - nsig_sr     #also during training
        init_Z = nsig_sr / np.sqrt(nbkg_sr) #Z0_train

        B_test = np.sum(labels == 0)

        Z0_val = nsig_sr / np.sqrt(B_test) # Z0 for changing nsigs, if we had test set with that nsig too
        bkg_preds = preds[labels == 0]
        threshold = np.quantile(bkg_preds, 1 - beff)

        Scut = np.sum((preds > threshold) & (labels != 0))

        seff = Scut / (np.sum(labels != 0) + 1e-10)
        sic = seff / np.sqrt(beff)

        Z = sic * init_Z  #projected Zcut

        results.setdefault(sig, {}).setdefault(nsig, []).append((Z, sic, Z0_val))

    return results


def find_pickles(name):
    # only the <Nsig> folders (numbers), not the CFM training job folder
    pickles = []
    for sig in LABEL_MAP:
        for path in glob(os.path.join(args.indir, sig, "*", "plots", "*_*", name)):
            if path.split(os.sep)[-4].isdigit():
                pickles.append(path)
    return pickles


def get_curves(results_dict, index=0):
    """index=0 for Z, index=1 for sic"""
    nsigs = sorted(results_dict.keys())
    lows, meds, highs = [], [], []

    for ns in nsigs:
        vals = np.array([v[index] for v in results_dict[ns]])
        q16, q50, q84 = np.quantile(vals, [0.16, 0.5, 0.84])
        lows.append(q16)
        meds.append(q50)
        highs.append(q84)

    return np.array(nsigs), (np.array(lows), np.array(meds), np.array(highs))


cathode_results = parse_all(find_pickles("CATHODEresults.pkl"))
iad_results = parse_all(find_pickles("IADresults.pkl"))

#extracting initial Z for corresponding nsig values
first_sig = next(iter(iad_results.values()))
nsig_vals = sorted(first_sig.keys())
z0_vals = [first_sig[ns][0][2] for ns in nsig_vals]  # first run, index 2 = Z0_val

# CATHODE
for sig, nsig_dict in cathode_results.items():
    nsig_cathode, (low_cathode, med_cathode, high_cathode) = get_curves(nsig_dict, index=0)
    color = COLOR_MAP.get(sig, None)
    plt.fill_between(nsig_cathode, low_cathode, high_cathode, alpha=0.4, color=color)
    plt.plot(nsig_cathode, med_cathode, label="CATHODE", color=color, linestyle='-', linewidth=3, alpha=0.8)

# IAD
for sig, nsig_dict in iad_results.items():
    nsig_iad, (low_iad, med_iad, high_iad) = get_curves(nsig_dict, index=0)
    label = LABEL_MAP.get(sig, sig)
    color = COLOR_MAP.get(sig, None)
    plt.plot(nsig_iad, med_iad, label=f"{label} IAD", color=color, linestyle='--', linewidth=3, alpha=1.0)

fig, ax = plt.gcf(), plt.gca()

# Interpolate nsig positions where Z0 equals 1, 2, 3, 4, ...
z0_arr = np.array(z0_vals)
nsig_arr = np.array(nsig_vals)
integer_z0s = np.arange(1, int(np.floor(z0_arr.max())) + 1)
nsig_at_integer_z0 = np.interp(integer_z0s, z0_arr, nsig_arr)

secax = ax.secondary_xaxis('top')
secax.set_xticks(nsig_at_integer_z0)
secax.set_xticklabels([str(int(z)) for z in integer_z0s])
secax.set_xlabel(r"$Z_0$", fontsize=16)
secax.tick_params(labelsize=16)

ax.text(
    0.18, 0.95,
    "Signal Region\n" + r"$3.3\text{–}3.7\ \mathrm{TeV}$",
    transform=ax.transAxes, ha='center', va='top', fontsize=14,
)

ax.set_xlabel("Signal injection", fontsize=16)
ax.set_ylabel(r"$Z_\mathrm{cut}$", fontsize=16)
ax.set_xlim(0, 3500)
ax.set_ylim(0, 25)
ax.tick_params(labelsize=14)

fig.subplots_adjust(top=0.88)
plt.savefig(args.output, bbox_inches='tight')
print(f"Saved {args.output}")
