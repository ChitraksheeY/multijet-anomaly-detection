"""Z_cut in the plane of two injected signals (N_sig1, N_sig2), for CATHODE (filled bands
and solid contours) and IAD (dashed contours), at a background efficiency of 1% (paper: Figure 6).

Uses the signal-mix results (training/multijet_SigMix_injection.sh) for the inner points and
the single-signal results (training/multijet_SingleSig_injection.sh) for the axes.

Usage:
  python plot_2Dscan_SignalMix.py --mixdir trainings/multijet/SigMix --singledir trainings/multijet/SingleSig \
      --output Z_CatIAD_2D_bands.pdf
"""
import os
import argparse
import pickle as pkl
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mixdir", required=True, help="Signal-mix results (<sig1>_x_<sig2>/<n1>_x_<n2>/plots/...)")
parser.add_argument("--singledir", required=True, help="Single-signal results (<sig>/<Nsig>/plots/...)")
parser.add_argument("--output", required=True, help="Output file (pdf)")
args = parser.parse_args()

signals = ["GKK_qq", "GKK_gR_Rgg_2217"]
show_bb3_label = True

# Second example pair
#signals = ["Wp_XY_X1000_Y1000", "GKK_gR_Rtt_500"]
#show_bb3_label = False

beff = 0.01  # background efficiency

LABEL_MAP = {
    "GKK_qq": r"$G_\mathrm{KK}\to qq$",
    "GKK_gR_Rgg_2217": r"$G_\mathrm{KK}\to gR$, $R\to gg$",
    "GKK_gR_Rgg_500": r"$R\to gg$, $m_R=500\ \mathrm{GeV}$",
    "GKK_gR_Rtt_2217": r"$R\to t\bar t$, $m_R=2217\ \mathrm{GeV}$",
    "GKK_gR_Rtt_500": r"$G_\mathrm{KK}\to gR$, $R\to t\bar t$, $m_R=500\ \mathrm{GeV}$",
    "Wp_XY_X500_Y100": r"$m_X=500, m_Y=100\ \mathrm{GeV}$",
    "Wp_XY_X1000_Y1000": r"$W'\to XY\to 4q$, $m_X=m_Y=1\ \mathrm{TeV}$",
    "Wp_XY_X1000_Y3000": r"$m_X=1, m_Y=3\ \mathrm{TeV}$",
    "Wp_XY_X2000_Y2000": r"$m_X=m_Y=2\ \mathrm{TeV}$",
}


def load_results(mode):
    pickles = list(glob(os.path.join(args.mixdir, f"{signals[0]}_x_{signals[1]}", "*", "plots", "*", f"{mode}results.pkl")))
    pickles += list(glob(os.path.join(args.singledir, f"{signals[0]}", "*", "plots", "*", f"{mode}results.pkl")))
    pickles += list(glob(os.path.join(args.singledir, f"{signals[1]}", "*", "plots", "*", f"{mode}results.pkl")))

    results = {}
    for pickle in tqdm(pickles, desc=mode):
        with open(pickle, "rb") as file:
            r = pkl.load(file)

        preds = r["preds"].flatten()
        labels = r["labels"]
        # signal names from the feature file names, e.g. .../GKK_qq.allfeatures.root -> GKK_qq
        sigs = [os.path.basename(s).split(".")[0] for s in r["sigs"]]
        nsig_sr = r["Nsig_SR_train"]  # during training

        if len(sigs) == 1 and sigs[0] == signals[0]:
            nsig = r["Nsig"] + [0]
        elif len(sigs) == 1 and sigs[0] == signals[1]:
            nsig = [0] + r["Nsig"]
        else:
            # They are in the right order if the folder naming convention is respected
            assert(sigs == signals)
            nsig = r["Nsig"]

        nbkg_sr = r["events_in_SR"] - nsig_sr
        init_Z = nsig_sr / np.sqrt(nbkg_sr)

        bkg_preds = preds[labels == 0]
        threshold = np.quantile(bkg_preds, 1 - beff)

        # The signals test sets do not follow the correct nsig ratios
        # So we need to reweight them
        weights = np.zeros_like(preds)  # background has zero weight
        for i, n in enumerate(r["Nsig"]):
            if n == 0:
                continue
            weights[labels == i + 1] = n / np.sum(labels == i + 1)

        # Sum of weights for signal events above threshold
        # Meaningless on its own
        S = np.sum(weights[(preds > threshold) & (labels != 0)])

        seff = S / (np.sum(weights) + 1e-10)
        sic = seff / np.sqrt(beff)

        Z = sic * init_Z  # corrected for training SR

        entry = results.get(tuple(nsig), [])
        entry.append((Z, sic))
        results[tuple(nsig)] = entry
    return results


def make_grid(results):
    x = np.array([n[0] for n in results])
    y = np.array([n[1] for n in results])
    z = np.array([np.median(r, axis=0) for r in results.values()])
    lut = LinearNDInterpolator(np.c_[x, y], z)
    X = np.linspace(min(x), max(x), 100)
    Y = np.linspace(min(y), max(y), 100)
    X, Y = np.meshgrid(X, Y)
    return X, Y, lut(X, Y)


iad_results     = load_results("IAD")
cathode_results = load_results("CATHODE")

X_iad,     Y_iad,     Z_iad     = make_grid(iad_results)
X_cat,     Y_cat,     Z_cat     = make_grid(cathode_results)


fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

sig_levels = [3, 5, 10]
max_Z = max(np.nanmax(Z_iad[..., 0]), np.nanmax(Z_cat[..., 0]))
fill_levels = [0, 3, 5, 10, max_Z]

# Teal palette light to dark for CATHODE filled bands
teal_colors = [
    "#ffffff",   # very light teal  (0–3 sigmas)
    "#e0f4f4",   # very light teal  (0–3 sigmas)
    "#7ecece",   # soft teal        (3–5)
    "#2a9d9d",   # medium teal      (5–10)
]
teal_line = "#1a6666"
purple    = "#6a3d9a"

cf = ax.contourf(
    X_cat, Y_cat, Z_cat[..., 0],
    levels=fill_levels,
    colors=teal_colors,
    alpha=0.33,
    zorder=1
)

# CATHODE solid contour lines on top of fill
for level, lw in zip(sig_levels, [1.5, 2.2, 3.0]):
    c = ax.contour(
        X_cat, Y_cat, Z_cat[..., 0],
        levels=[level],
        colors=[teal_line],
        linewidths=lw,
        linestyles="solid",
        zorder=3
    )
    ax.clabel(c, fmt=r"$%d\,\sigma$", fontsize=12, inline=True, zorder=4)

#for IAD
for level, lw in zip(sig_levels, [1.5, 2.2, 3.0]):
    c = ax.contour(
        X_iad, Y_iad, Z_iad[..., 0],
        levels=[level],
        colors=[purple],
        linewidths=lw,
        linestyles="dashed",
        zorder=3
    )
    ax.clabel(c, fmt=r"$%d\,\sigma$", fontsize=12, inline=True, zorder=4)

# SIC = 1 dotted gray
ax.contour(
    X_iad, Y_iad, Z_iad[..., 1],
    levels=[1],
    colors=["#888888"],
    linewidths=1.0,
    linestyles="dotted",
    zorder=3
)


class HandlerFilledLine(HandlerBase):
    def __init__(self, color_fill, color_line, alpha=0.33, lw=2.0):
        self.color_fill = color_fill
        self.color_line = color_line
        self.alpha      = alpha
        self.lw         = lw
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        patch = mpatches.FancyBboxPatch(
            [xdescent, ydescent], width, height,
            boxstyle="square,pad=0",
            facecolor=self.color_fill, alpha=self.alpha,
            edgecolor="none", transform=trans
        )
        line = plt.Line2D(
            [xdescent, xdescent + width],
            [ydescent + height / 2, ydescent + height / 2],
            color=self.color_line, linewidth=self.lw, transform=trans
        )
        return [patch, line]


method_handles = [
    Patch(facecolor="#7ecece", alpha=0.55, label="CATHODE"),
    Line2D([0], [0], color=purple, linewidth=2, linestyle="dashed", label="IAD"),
    Line2D([0], [0], color="#777676", linewidth=1.0, linestyle="dotted", label="SIC $= 1$"),
]

ax.set_xlim(0, 1750)
ax.set_ylim(0, 2500)
ax.set_xlabel(f"$N_{{\\mathrm{{sig1}}}}$ ({LABEL_MAP.get(signals[0], signals[0])})", fontsize=16)
ax.set_ylabel(f"$N_{{\\mathrm{{sig2}}}}$ ({LABEL_MAP.get(signals[1], signals[1])})", fontsize=16)
ax.tick_params(axis='both', labelsize=16)

if show_bb3_label:
    ax.scatter([1200], [2000], s=100, marker="+", color="black", zorder=5)
    ax.annotate("Black Box 3", xy=(1200, 2000), xycoords='data',
                xytext=(0, -10), textcoords='offset points',
                ha='center', va='top', fontsize=14)

leg = ax.legend(handles=method_handles,
    handler_map={method_handles[0]: HandlerFilledLine("#7ecece", teal_line, alpha=0.33, lw=2.0)},
    frameon=False, fontsize=14, loc = "upper right")
leg.get_frame().set_facecolor("white")
leg.get_frame().set_alpha(0.5)
leg.get_frame().set_edgecolor("none")

plt.tight_layout()
plt.savefig(args.output, dpi=300)
print(f"Saved {args.output}")
