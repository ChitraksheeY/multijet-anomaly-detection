"""R&D validation with M_RSD as resonant variable (paper: Figure 7, right),
and the shared legend panel for both plots of Figure 7.

Z_cut vs. number of injected signal events for CATHODE (solid line, 16-84% band)
and IAD (dashed line), at a background efficiency of 1%, for M_RSD + event-level
features and M_RSD + dijet features.

Reads the results of training/rnd_injection.sh (COMBO=mrsd_HTtaus and COMBO=mrsd_Mjtaus).

Usage:
  python plot_1D_RnD_Mrsd.py --indir trainings/rnd --output RnDwithMrsd.pdf --legend-output legend_panel_signals.pdf
"""
import os
import argparse
import pickle as pkl
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--indir", required=True, help="Directory with one folder per COMBO (<COMBO>/<Nsig>/plots/...)")
parser.add_argument("--output", required=True, help="Output file (pdf)")
parser.add_argument("--legend-output", required=True, help="Output file for the shared legend panel (pdf)")
args = parser.parse_args()

beff = 0.01  # background efficiency

LABEL_MAP = {
    "mrsd_HTtaus": r"$M_{\mathrm{RSD}}$ + event-level features",
    "mrsd_Mjtaus": r"$M_{\mathrm{RSD}}$ + dijet features",
    "mjj_HTtaus":  r"$M_{jj}$ + event-level features",
    "mjj_Mjtaus": r"$M_{jj}$ + dijet features",
}

COLOR_MAP = {
    "mrsd_HTtaus": "#8B2D3D",
    "mrsd_Mjtaus": "#D4994A",
    "mjj_HTtaus":  "#7B2D8B",
    "mjj_Mjtaus":  "#117A7A",
}

# This plot shows the M_RSD combinations; the M_jj ones are only used for the legend panel
plot_keys = ["mrsd_HTtaus", "mrsd_Mjtaus"]


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
    for sig in plot_keys:
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
    "Signal Region\n" + r"$2.7\text{–}3.7\ \mathrm{TeV}$",
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

# --- Shared legend panel for both plots of Figure 7 ---
# Left plot = Mjj group
left_keys = ["mjj_HTtaus", "mjj_Mjtaus"]

# Right plot = Mrsd group
right_keys = ["mrsd_HTtaus", "mrsd_Mjtaus"]

signal_handles_left = [
    Line2D([0], [0], color=COLOR_MAP[sig], linewidth=3, label=LABEL_MAP[sig])
    for sig in left_keys
]

signal_handles_right = [
    Line2D([0], [0], color=COLOR_MAP[sig], linewidth=3, label=LABEL_MAP[sig])
    for sig in right_keys
]

# Shared CATHODE / IAD handles — appear only once
method_handles = [
    Line2D([0], [0], color='black', linestyle='-',  linewidth=3, label='CATHODE'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=3, label='IAD'),
]

# Standalone wide figure — width = 2x a single plot's width
fig_leg = plt.figure(figsize=(16, 2))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')

# Top row: signal colors, left group + right group side by side
leg1 = ax_leg.legend(
    handles=signal_handles_left + signal_handles_right,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,                     # 2 columns so left-group / right-group line up visually
    frameon=False,
    fontsize=16,
    handlelength=2.5,
    columnspacing=2.5,
)
ax_leg.add_artist(leg1)

# Bottom row: CATHODE / IAD, single shared row
leg2 = ax_leg.legend(
    handles=method_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.55),
    ncol=2,
    frameon=False,
    fontsize=16,
    handlelength=2.5,
)

fig_leg.savefig(
    args.legend_output,
    bbox_inches="tight",
    pad_inches=0.2,
    dpi=300,
)
plt.close(fig_leg)
print(f"Saved {args.legend_output}")
