"""M_JJ, M_all and M_RSD side by side for a set of signals (paper: Figures 1 and 2).

Reads the first 3200 events of each signal feature file (<name>.allfeatures.root).

Usage:
  python plot_masses.py --datadir /path/to/feature_files --set bb3   --output allmasses_bb3.pdf
  python plot_masses.py --datadir /path/to/feature_files --set other --output allmasses_sidebyside.pdf
"""
import argparse
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import uproot as up
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--datadir", required=True, help="Directory with the <name>.allfeatures.root files")
parser.add_argument("--set", required=True, choices=["bb3", "other"], help="Which signals to show")
parser.add_argument("--output", required=True, help="Output file (pdf)")
args = parser.parse_args()

if args.set == "bb3":
    LABEL_MAP = {
        "GKK_qq": r"$G_\mathrm{KK}\to qq$",
        "GKK_gR_Rgg_2217": r"$G_\mathrm{KK}\to gR \to ggg$" + r", $m_R=2217\ \mathrm{GeV}$",
    }
else:
    LABEL_MAP = {
        "GKK_gR_Rgg_500": r"$G_\mathrm{KK}\to gR, R \to gg$, $m_R=500\ \mathrm{GeV}$",
        "GKK_gR_Rtt_2217": r"$G_\mathrm{KK}\to gR, R\to t\bar t$, $m_R=2217\ \mathrm{GeV}$",
        "Wp_XY_X1000_Y1000": r"$W'\to XY\to 4q$" + r", $m_X=m_Y=1\ \mathrm{TeV}$",
    }

COLOR_MAP = {
    "GKK_qq": "#1b6599",
    "GKK_gR_Rgg_2217": "#00a7b9",
    "GKK_gR_Rgg_500": "#2ca02c",
    "GKK_gR_Rtt_500": "#9467bd",
    "GKK_gR_Rtt_2217": "#0d4266",
    "Wp_XY_X500_Y100": "#d62728",
    "Wp_XY_X1000_Y1000": "#ff7f0e",
    "Wp_XY_X1000_Y3000": "#e377c2",
    "Wp_XY_X2000_Y2000": "#8c564b",
}


def load_masses(path):
    """Columns: M_JJ, M_all, M_RSD (GeV)."""
    with up.open(path) as f:
        tree = f["lhco_blackbox"]
        mjj = tree["mj1j2"].array(library="np")
        m_all = tree["m_all"].array(library="np")
        m_rsd = tree["m_rsd"].array(library="np")
    return np.hstack((mjj[:, np.newaxis], m_all[:, np.newaxis], m_rsd[:, np.newaxis]))


signals = list(LABEL_MAP.keys())
data = {
    sig: load_masses(join(args.datadir, f"{sig}.allfeatures.root"))[:3200]
    for sig in signals
}

nbins = 100
range_ = (1.0, 10.0)  # TeV
bins = np.linspace(range_[0], range_[1], nbins + 1)

mass_map = {
    0: {"col": 0, "label": r"$M_\mathrm{JJ}\ \mathrm{[TeV]}$"},
    1: {"col": 1, "label": r"$M_\mathrm{all}\ \mathrm{[TeV]}$"},
    2: {"col": 2, "label": r"$M_\mathrm{RSD}\ \mathrm{[TeV]}$"},
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.set_facecolor("white")
plt.rcdefaults()

handles = []

for i, (mass, info) in enumerate(mass_map.items()):
    ax = axes[i]
    col = info["col"]

    for sig in signals:
        ax.hist(
            data[sig][:, col] / 1000.0,   # GeV to TeV
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color=COLOR_MAP.get(sig, "black"),
        )
        if i == 0:
            handles.append(
                Line2D([0], [0], color=COLOR_MAP.get(sig, "black"),
                       lw=1.8, label=LABEL_MAP.get(sig, sig))
            )

    ax.set_yscale("log")
    ax.set_xlim(range_)
    ax.set_xlabel(info["label"], fontsize=18)

    # Major ticks on all sides, minor ticks only bottom/left
    ax.tick_params(axis="both", which="major", labelsize=16, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", length=3, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

# --- Vertical reference lines (signal region) on the last panel only ---
for x in [3, 4]:
    axes[-1].axvline(x=x, linestyle="--", color="0.3", linewidth=1.5)

axes[0].set_ylabel("Events", fontsize=18)

leg = fig.legend(
    handles=handles,
    fontsize=18,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=len(handles),
    columnspacing=1.8,
)

plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight")
print(f"Saved {args.output}")
