"""Input features in the signal region (M_RSD 3-4 TeV): background, generated (CATHODE)
samples and three signals, with (X - Bkg)/sqrt(Bkg) in the lower panel (paper: Figure 4).
One file per feature (sr_genVsSig_var<col>.pdf) plus a separate legend panel (legend_panel.pdf).

Inputs:
  --datadir  directory with the feature files: the data file and <signal>.allfeatures.root
  --samples  CATHODE_SR_samples.npy written by training/train.py --save-samples
             (e.g. from training/multijet_train_cfm.sh)

Usage:
  python plot_features_SR.py --datadir /path/to/feature_files \
      --samples trainings/multijet/cfm_job/plots/<...>/CATHODE_SR_samples.npy --outdir Features/
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import uproot as up
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--datadir", required=True, help="Directory with the feature files")
parser.add_argument("--data", default="events_LHCO2025_BlackBox3.allfeatures.root", help="Data file (background = type 0)")
parser.add_argument("--samples", required=True, help="CATHODE_SR_samples.npy from train.py --save-samples")
parser.add_argument("--outdir", required=True, help="Output directory")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

branches = ["m_rsd", "HT", "tau1", "tau2", "tau3", "tau4"]
SIGNALS = ["GKK_qq", "GKK_gR_Rgg_2217", "Wp_XY_X1000_Y1000"]


def load(path, branches):
    """Loads the requested features and the type branch (last column) from a tree."""
    with up.open(path) as f:
        tree = f["lhco_blackbox"]
        events = [tree[b].array(library="np").reshape((-1, 1)) for b in branches]
        types = tree["type"].array(library="np").reshape((-1, 1))
    return np.hstack((*events, types))


data = load(os.path.join(args.datadir, args.data), branches)
bkg = data[data[:, -1] == 0, :-1]
sig1, sig2, sig3 = [load(os.path.join(args.datadir, f"{s}.allfeatures.root"), branches)[:, :-1] for s in SIGNALS]
samples = np.load(args.samples)   # generated SR samples, GeV


def to_sr_tev(arr):
    arr_sr = arr[(arr[:, 0] >= 3000) & (arr[:, 0] <= 4000)].copy()
    return arr_sr / 1000.0

bkg_sr = to_sr_tev(bkg)
sig1_sr = to_sr_tev(sig1)
sig2_sr = to_sr_tev(sig2)
sig3_sr = to_sr_tev(sig3)

# Samples are already all in the SR
samples_sr = samples / 1000.0   # scale to TeV

N_SIG_INJECT = 2000   # number of signal events shown for each signal
RNG          = np.random.default_rng(seed=42)   # fixed seed for reproducibility

sig1_sr = sig1_sr[:N_SIG_INJECT]
sig2_sr = sig2_sr[:N_SIG_INJECT]
sig3_sr = sig3_sr[:N_SIG_INJECT]

# Generated samples: same number of events as background + one injected signal
n_target    = len(bkg_sr) + N_SIG_INJECT
idx_sam     = RNG.choice(len(samples_sr), size=n_target, replace=False)
samples_sr  = samples_sr[idx_sam]

print(f"Background in SR        : {len(bkg_sr):>8,} events")
print(f"Signal events shown     : {N_SIG_INJECT:>8,} events")
print(f"Generated (subsampled)  : {len(samples_sr):>8,} events")

C_BKG  = "#7C7C7C"
C_GEN  = "#3B3A3A"
C_SIG1 = "#186aa5"
C_SIG2 = "#51bdc9"
C_SIG3 = "#ff7f0e"

SIG_LABELS = (r"$G_\mathrm{KK}\to qq$", r"$G_\mathrm{KK}\to gR, R\to gg$", r"$W'\to XY\to 4q, m_X=m_Y=1\ \mathrm{TeV}$")
SIG_COLORS = (C_SIG1, C_SIG2, C_SIG3)

plt.rcdefaults()


def raw_counts(vals, bins):
    counts = np.histogram(vals, bins=bins)[0].astype(float)
    err    = np.sqrt(counts)
    return counts, err


def sig_over_sqrtbkg_raw(sig_vals, bkg_vals, bins):
    """sig / sqrt(bkg), propagated with derivatives, raw counts."""
    sig_counts, sig_err = raw_counts(sig_vals, bins)
    bkg_counts, bkg_err = raw_counts(bkg_vals, bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        val      = np.where(bkg_counts > 0, sig_counts / np.sqrt(bkg_counts), np.nan)
        term_sig = np.where(bkg_counts > 0, sig_err / np.sqrt(bkg_counts), np.nan)
        term_bkg = np.where(bkg_counts > 0,
                            0.5 * sig_counts / bkg_counts * bkg_err / np.sqrt(bkg_counts),
                            np.nan)
        err = np.sqrt(term_sig**2 + term_bkg**2)
    return val, err


def pull_raw(sam_vals, bkg_vals, bins):
    """(sam - bkg) / sqrt(bkg), propagated with derivatives, raw counts."""
    sam_counts, sam_err = raw_counts(sam_vals, bins)
    bkg_counts, bkg_err = raw_counts(bkg_vals, bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        val      = np.where(bkg_counts > 0, (sam_counts - bkg_counts) / np.sqrt(bkg_counts), np.nan)
        term_sam = np.where(bkg_counts > 0, sam_err / np.sqrt(bkg_counts), np.nan)
        term_bkg = np.where(bkg_counts > 0,
                            np.abs(- 0.5 * ((sam_counts + bkg_counts) / bkg_counts**(1.5)) * bkg_err),
                            np.nan)
        err = np.sqrt(term_sam**2 + term_bkg**2)
    return val, err


def draw_cms_points(ax, bins, val, err, color, lw, label=None):
    """CMS-style: one point + vertical error bar per bin, centred in bin."""
    centres = 0.5 * (bins[:-1] + bins[1:])
    ax.errorbar(
        centres, val, yerr=err,
        fmt="o", ms=3.5, lw=lw, capsize=2, capthick=lw,
        color=color, label=label, zorder=3,
    )


VARIABLES = [
    dict(col=1, xlabel=r"$H_\mathrm{T}$ [TeV]", show_legend=False,  xlim=(bkg_sr[:, 1].min(), None)),
    dict(col=2, xlabel=r"$\tau_1$ [TeV]",        show_legend=False, xlim=(bkg_sr[:, 2].min(), None)),
    dict(col=3, xlabel=r"$\tau_2$ [TeV]",        show_legend=False, xlim=(bkg_sr[:, 3].min(), None)),
    dict(col=4, xlabel=r"$\tau_3$ [TeV]",        show_legend=False, xlim=(bkg_sr[:, 4].min(), None)),
    dict(col=5, xlabel=r"$\tau_4$ [TeV]",        show_legend=False, xlim=(bkg_sr[:, 5].min(), None)),
]


def make_bins(*arrays, n_bins=50):
    lo = min(a.min() for a in arrays)
    hi = max(a.max() for a in arrays)
    return np.linspace(lo, hi, n_bins + 1)


for var in VARIABLES:
    col         = var["col"]
    xlabel      = var["xlabel"]
    show_legend = var["show_legend"]
    xlim        = var["xlim"]

    bkg_vals  = bkg_sr[:, col]
    sam_vals  = samples_sr[:, col]
    sig_vals  = [sig1_sr[:, col], sig2_sr[:, col], sig3_sr[:, col]]

    bins = make_bins(bkg_vals, sam_vals, *sig_vals)

    bkg_raw, bkg_err = raw_counts(bkg_vals, bins)
    sam_raw, sam_err = raw_counts(sam_vals, bins)
    sig_raws_errs    = [raw_counts(sv, bins) for sv in sig_vals]

    # ── Figure layout: 1 main + 1 combined panel ──────────────────────────
    height_ratios = [3, 2]
    fig = plt.figure(figsize=(8, 5 + 2.0))
    gs  = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.0)

    ax_main  = fig.add_subplot(gs[0])
    ax_lower = fig.add_subplot(gs[1], sharex=ax_main)

    # ── Main panel ────────────────────────────────────────────────────────
    ax_main.step(bins[:-1], bkg_raw, where="post",
                color=C_BKG, linewidth=1.8, label="Background", zorder=3)
    ax_main.fill_between(bins[:-1], bkg_raw, step="post",
                        alpha=0.20, color=C_BKG, zorder=2)
    ax_main.step(bins[:-1], sam_raw, where="post",
                color=C_GEN, linewidth=1.8, label="Generated", zorder=4)

    for (sr, _), color, sig_label in zip(sig_raws_errs, SIG_COLORS, SIG_LABELS):
        ax_main.step(bins[:-1], sr, where="post",
                    color=color, linewidth=2.1, label=sig_label, zorder=5)

    ax_main.set_xlim(*xlim)
    ax_main.set_yscale("log")
    ax_main.tick_params(axis="both", labelsize=16)
    ax_main.yaxis.set_label_coords(-0.12, 0.5)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    if show_legend:
        ax_main.legend(fontsize=16, handlelength=1.5)

    # ── Lower panel: all significances overlaid, raw counts ───────────────
    for sv, color, sig_label in zip(sig_vals, SIG_COLORS, SIG_LABELS):
        val, err = sig_over_sqrtbkg_raw(sv, bkg_vals, bins)
        draw_cms_points(ax_lower, bins, val, err, color=color, lw=1.2, label=sig_label)

    val_pull, err_pull = pull_raw(sam_vals, bkg_vals, bins)
    draw_cms_points(ax_lower, bins, val_pull, err_pull, color=C_GEN, lw=1.4, label="Generated")

    ax_lower.axhline(0.0, color="black", lw=0.8, linestyle="--", zorder=1)
    ax_lower.set_xlim(*xlim)
    ax_lower.tick_params(axis="both", labelsize=16)
    ax_lower.set_xlabel(xlabel, fontsize=16)

    fig.align_ylabels([ax_main, ax_lower])
    plt.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.14)

    fig.canvas.draw()

    fig.text(
        0.055, 0.65,
        "Events",
        va="center",
        ha="center",
        fontsize=15,
        rotation=90
    )

    # Lower-panel y-label
    fig.text(
        0.055, 0.35,
        r"$(X-\mathrm{Bkg})/\sqrt{\mathrm{Bkg}}$"
        ", "
        r"$X=\mathrm{Gen},\,\mathrm{Bkg}+Sig$",
        va="center",
        ha="center",
        fontsize=15,
        rotation=90
    )

    plt.savefig(os.path.join(args.outdir, f"sr_genVsSig_var{col}.pdf"), dpi=300)
    plt.close()


# --- Separate legend panel ---
legend_handles = [
    Line2D([0], [0], color=C_BKG, linewidth=2.1, label="Background"),
    Line2D([0], [0], color=C_GEN, linewidth=2.1, label="Generated"),
] + [
    Line2D([0], [0], color=color, linewidth=2.1, label=sig_label)
    for color, sig_label in zip(SIG_COLORS, SIG_LABELS)
]

fig_leg = plt.figure(figsize=(8, 5 + 2.0))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')

leg = ax_leg.legend(
    handles=legend_handles,
    loc='center',
    frameon=False,
    fontsize=17,
    handlelength=2.5,
    labelspacing=1.2,
)

fig_leg.savefig(os.path.join(args.outdir, "legend_panel.pdf"), bbox_inches="tight", dpi=300,)
plt.close(fig_leg)
print(f"Saved plots to {args.outdir}")
