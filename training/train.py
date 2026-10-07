#!/usr/bin/env python
"""Train supervised, IAD and CATHODE classifiers in a signal region.

Inputs are ROOT files written by compute_features.run (tree "lhco_blackbox").
The "type" branch is 0 for background and 1 for signal.

  --data      data events; only background (type == 0) is kept, signal is injected from --sigs
  --sigs      signal files, one --Nsig per file; 1000 events per signal are kept for testing
  --extrabkg  extra background for the IAD (idealised anomaly detector) and for all evaluation

Results (labels and classifier outputs, etc on test set) are saved as .pkl files for the plotting scripts.

Example:
  python train.py --data events_LHCO2025_BlackBox3.allfeatures.root \
      --sigs events_BB3_KKg2qq.allfeatures.root --Nsig 1000 \
      --extrabkg events_LHCO2025_BlackBox3_extrabkg.allfeatures.root --exBkg 1000000 \
      --min 3000 --max 4000 --mass m_rsd --features HTtaus --outdir out --run 0
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle as pkl
import uproot as up
from datetime import datetime

from os.path import exists, join
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sk_cathode.generative_models.conditional_flow_matching import ConditionalFlowMatching
from sk_cathode.classifier_models.neural_network_classifier import NeuralNetworkClassifier
from sk_cathode.classifier_models.boosted_decision_tree import HGBClassifier
from sk_cathode.utils.preprocessing import LogitScaler


# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Train supervised, IAD and CATHODE classifiers")
parser.add_argument("--data", type=str, required=True, help="Data ROOT file (background = type 0)")
parser.add_argument("--sigs", type=str, nargs="*", default=[], help="Signal ROOT files")
parser.add_argument("--Nsig", type=int, nargs="*", default=[], help="Number of injected signal events, one per signal file")
parser.add_argument("--extrabkg", type=str, nargs="+", required=True, help="Extra background ROOT file(s)")
parser.add_argument("--exBkg", type=int, required=True, help="Number of extra background events for IAD training")
parser.add_argument("--min", type=float, required=True, help="MinMass SR")
parser.add_argument("--max", type=float, required=True, help="MaxMass SR")
parser.add_argument("--mass", default="m_rsd", choices=["m_rsd", "mj1j2"], help="Mass definition")
parser.add_argument("--features", default="HTtaus", choices=["HTtaus", "Mjtaus"], help="Feature set")
parser.add_argument("--outdir", type=str, required=True, help="Output directory for trained models")
parser.add_argument("--pkldir", type=str, default=None, help="Directory for pkl results and plots (default: outdir)")
parser.add_argument("--flowdir", type=str, default=None, help="CFM directory (default: inside outdir)")
parser.add_argument("--run", type=int, default=0, help="Run number")
parser.add_argument("--arch", default="NN", choices=["NN", "BDT"], help="Classifier architecture")
parser.add_argument("--batch-size", default=1024, type=int, help="IAD and CATHODE classifier batch size")
parser.add_argument("--cfm-epochs", default=30000, type=int, help="CFM training epochs")
parser.add_argument("--skip-sup", default=False, action="store_true", help="Do not train a supervised classifier")
parser.add_argument("--skip-iad", default=False, action="store_true", help="Do not train an IAD classifier")
parser.add_argument("--skip-cathode", default=False, action="store_true", help="Do not run CATHODE (CFM + classifier)")
parser.add_argument("--skip-cfm-training", default=False, action="store_true",
                    help="Do not train the CFM, load the last saved epoch from --flowdir")
parser.add_argument("--save-samples", default=False, action="store_true",
                    help="Save the generated SR samples (mass, features) in GeV as CATHODE_SR_samples.npy")

args = parser.parse_args()

if len(args.sigs) != len(args.Nsig):
    parser.error("give one --Nsig per signal file")

print("Signals:", ", ".join(args.sigs))
print("Counts:", ", ".join(map(str, args.Nsig)))
print(f"Signal region: {args.min} to {args.max}")
print(f"Extra background for IAD training: {args.exBkg}")
print(f"Output directory: {args.outdir}")

# dedicated directory for results and plots
date_str = datetime.now().strftime("%d%B%Y")
pkl_dir = join(args.pkldir or args.outdir, "plots", f"{args.mass}_{args.features}_SR{args.min}to{args.max}_{args.run}_{date_str}")
os.makedirs(pkl_dir, exist_ok=True)

# Models
flow_savedir = args.flowdir or join(args.outdir, f"trained_CFM_SR{args.min}to{args.max}")
iad_classifier_savedir = join(args.outdir, f"trained_IADclsf_SR{args.min}to{args.max}_{args.run}")
ca_classifier_savedir = join(args.outdir, f"trained_CATHODEclsf_SR{args.min}to{args.max}_{args.run}")
sup_clsf_savedir = join(args.outdir, f"trained_supclsf_SR{args.min}to{args.max}_{args.run}")

# Determine branches to load
# Feature 0 is always the mass
branches = [args.mass]
# Input features
if args.features == "HTtaus":
    branches += ["HT", "tau1", "tau2", "tau3", "tau4"]
elif args.features == "Mjtaus":
    branches += ["mjmin", "delta_mjj", "tau21min", "tau21max"]


def load(path, branches, label):
    """
    Loads the requested features from a tree.
    The last column is the "type" branch if present, otherwise `label`.
    """
    with up.open(path) as f:
        events = [
            f["lhco_blackbox"][b].array(library="np").reshape((-1, 1))
            for b in branches
        ]
        if "type" in f["lhco_blackbox"].keys():
            labels = f["lhco_blackbox"]["type"].array(library="np").reshape((-1, 1))
        else:
            labels = np.full((len(events[0]), 1), label) 
    return np.hstack((*events, labels))


# Data: keep only background, signal is injected below
data = load(args.data, branches, 0)
bkg = data[data[:, -1] == 0] # for cases where dataset has both signal and background

# Signals: keep type != 0 and label them 1, 2, ... (one label per signal file)
full_sigs = []
for i, path in enumerate(args.sigs):
    sig = load(path, branches, 1)
    sig = sig[sig[:, -1] != 0]
    sig[:, -1] = i + 1
    full_sigs.append(sig)

# Extra background is background by construction
full_extrabkg = np.concatenate([load(path, branches, 0) for path in args.extrabkg], axis=0)
full_extrabkg[:, -1] = 0

print(f"Using {len(branches)} features:", ", ".join(branches))
#------------------------------------------------------------------------------------

# Plot all features
plt.figure(figsize=(15, 3 * (len(branches) + 2) // 3))
for i, name in enumerate(branches):
    plt.subplot((len(branches) + 2) // 3, 3, i+1)
    _, bins, _ = plt.hist(bkg[:, i], bins=50, density=True, alpha=0.5, color="lightgray", label="Bkg")
    for sig, label in zip(full_sigs, args.sigs):
        plt.hist(sig[:, i], bins=bins, density=True, histtype="step", alpha=0.5, label=os.path.basename(label))
    plt.hist(full_extrabkg[:, i], bins=bins, density=True, histtype="step", color="black", linewidth=1.2, label="ExtraBkg")
    plt.yscale("log")
    plt.title(name)
    plt.xlabel(name)
    plt.ylabel("Arbitrary units")
    plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(join(pkl_dir, "all_features.pdf"))
plt.close()

#------------------------------------------------------------------------------------

def pick(events, n_train, n_test):
    if len(events) < n_train + n_test:
        raise ValueError(f"Dataset is too small! Want {n_train+n_test} and got {len(events)}")

    # Take the test set at the beginning so it doesn't change when we change n_train
    return events[n_test:n_test + n_train], events[:n_test]


# 1k for testing, for each signal
sig_train = []
sig_val = []
sig_test = []
for sig, n in zip(full_sigs, args.Nsig):
    train_val, test = pick(sig, n, 1000)
    if n > 0:
        train, val = train_test_split(train_val, test_size=0.1, random_state=42)
    else:
        train, val = train_val, train_val  # train_val is empty anyway but it has the correct shape

    sig_train.append(train)
    sig_val.append(val)
    sig_test.append(test)

bkg_train, bkg_val = train_test_split(bkg, test_size=0.10, random_state=42)

#1M for testing
extrabkg, exbkg_test = pick(full_extrabkg, args.exBkg, 1_000_000)
extrabkg_train, extrabkg_val = train_test_split(extrabkg, test_size=0.10, random_state=42)

print("=== Training set shapes ===")
print("(label included)")
print("bkg :", bkg_train.shape)
print("sigs:", [s.shape for s in sig_train])
print("extrabkg for IAD:", extrabkg_train.shape)
print()
print("=== Validation set shapes ===")
print("(label included)")
print("bkg :", bkg_val.shape)
print("sigs:", [s.shape for s in sig_val])
print("extrabkg for IAD:", extrabkg_val.shape)
print()
print("=== Test set shapes ===")
print("(label included)")
print("sigs:", [s.shape for s in sig_test])
print("extrabkg:", exbkg_test.shape)
print()

#Combine the sets and shuffle
data_train = np.vstack([bkg_train] + sig_train)
data_val = np.vstack([bkg_val] + sig_val)
data_test = np.vstack([exbkg_test] + sig_test)

data_train = shuffle(data_train, random_state=42)
data_val = shuffle(data_val, random_state=42)
data_test = shuffle(data_test, random_state=42)

print(f"data_train: {data_train.shape}")
print(f"data_val  : {data_val.shape}")
print(f"data_test : {data_test.shape}")


#data preparation
def separate_SB_SR(dataset, minmass, maxmass, min_sideband=1400.0):
    mass_column = dataset[:, 0]
    # Signal Region: strictly between minmass and maxmass
    innermask = (mass_column > minmass) & (mass_column < maxmass)

    # Sideband Region: mass >= min_sideband, but outside the signal region
    outermask = ((mass_column >= min_sideband) & ~innermask)

    return dataset[innermask], dataset[outermask]

# Define our signal region
innerdata_train, outerdata_train = separate_SB_SR(data_train, args.min, args.max)
innerdata_val, outerdata_val = separate_SB_SR(data_val, args.min, args.max)
innerdata_test, outerdata_test = separate_SB_SR(data_test, args.min, args.max)

print("=== data (SR only) ===")
print(f"innerdata_train: {innerdata_train.shape}")
print(f"innerdata_val: {innerdata_val.shape}")
print(f"innerdata_test: {innerdata_test.shape}")
print()
print("=== data (SB only) ===")
print(f"outerdata_train: {outerdata_train.shape}")
print(f"outerdata_val: {outerdata_val.shape}")
print(f"outerdata_test: {outerdata_test.shape}")


# Split extraBkg_IAD into SR and SB
SR_extraBkg_IAD_train, _ = separate_SB_SR(extrabkg_train, args.min, args.max)
SR_extraBkg_IAD_val, _ = separate_SB_SR(extrabkg_val, args.min, args.max)

print(f"SR_extraBkg_IAD_train: {SR_extraBkg_IAD_train.shape}")
print(f"SR_extraBkg_IAD_val: {SR_extraBkg_IAD_val.shape}")


# For SUPERVISED CLSF
# bkg=0
innerdata_bkg_train = innerdata_train[innerdata_train[:, -1] == 0]
innerdata_bkg_val = innerdata_val[innerdata_val[:, -1] == 0]
innerdata_bkg_test = innerdata_test[innerdata_test[:, -1] == 0]
# sig=1
innerdata_sig_train = innerdata_train[innerdata_train[:, -1] != 0]
innerdata_sig_val = innerdata_val[innerdata_val[:, -1] != 0]
innerdata_sig_test = innerdata_test[innerdata_test[:, -1] != 0]


def make_classifier(**kwargs):
    if args.arch == "NN":
        return NeuralNetworkClassifier(**kwargs)
    elif args.arch == "BDT":
        # Remove NN-specific arguments first
        for nnarg in ("n_inputs", "epochs", "batch_size"):
            if nnarg in kwargs:
                del kwargs[nnarg]
        return HGBClassifier(**kwargs)


if len(innerdata_sig_train) == 0:
    print("No signal. Skipping supervised classifier")
elif args.skip_sup:
    print("Skipping supervised classifier.")
else:
    print("=== For SUPERVISED CLSF ===")
    print(f"innerdata_bkg_train: {innerdata_bkg_train.shape}")
    print(f"innerdata_bkg_val: {innerdata_bkg_val.shape}")
    print(f"innerdata_bkg_test: {innerdata_bkg_test.shape}")
    print()
    print(f"innerdata_sig_train: {innerdata_sig_train.shape}")
    print(f"innerdata_sig_val: {innerdata_sig_val.shape}")
    print(f"innerdata_sig_test: {innerdata_sig_test.shape}")

    #supervised classifier
    sup_clsf_train_set = np.vstack([innerdata_bkg_train,innerdata_sig_train])
    sup_clsf_val_set = np.vstack([innerdata_bkg_val,innerdata_sig_val])

    sup_clsf_train_set = shuffle(sup_clsf_train_set, random_state=42)
    sup_clsf_val_set = shuffle(sup_clsf_val_set, random_state=42)

    sup_scaler = StandardScaler()
    sup_scaler.fit(sup_clsf_train_set[:, 1:-1])

    sup_X_train = sup_scaler.transform(sup_clsf_train_set[:, 1:-1])
    sup_y_train = sup_clsf_train_set[:, -1]
    sup_X_val = sup_scaler.transform(sup_clsf_val_set[:, 1:-1])
    sup_y_val = sup_clsf_val_set[:, -1]
    # Forget the difference between signals
    sup_y_train[sup_y_train != 0] = 1
    sup_y_val[sup_y_val != 0] = 1


    def train():
        sup_classifier_model = make_classifier(save_path=sup_clsf_savedir,
                                               n_inputs=sup_X_train.shape[1],
                                               early_stopping=True,
                                               epochs=None,
                                               verbose=True)
        sup_classifier_model.fit(sup_X_train, sup_y_train, sup_X_val, sup_y_val)


    # Let's protect ourselves from accidentally overwriting a trained model.
    if not exists(join(sup_clsf_savedir, "CLSF_val_losses.npy")):
        train()
    else:
        print(f"The model exists already in {sup_clsf_savedir}. Remove first if you want to overwrite.")

    try:
        sup_classifier_model = make_classifier(save_path=sup_clsf_savedir,
                                               n_inputs=sup_X_train.shape[1],
                                               load=True)
    except FileNotFoundError:
        # Sometimes the models are corrupted (e.g., after preemption)
        train()
        sup_classifier_model = make_classifier(save_path=sup_clsf_savedir,
                                               n_inputs=sup_X_train.shape[1],
                                               load=True)


    # Lets predeict and evaluate its performance
    sup_X_test = sup_scaler.transform(innerdata_test[:, 1:-1])
    # now let's evaluate the signal extraction performance
    sup_preds_test = sup_classifier_model.predict(sup_X_test)

    results = {
        "mode": "Supervised",
        "sigs": args.sigs,
        "Nsig": args.Nsig,
        "min": args.min,
        "max": args.max,
        "exBkg": args.exBkg,
        "Nsig_SR_train": int(len(innerdata_sig_train) + len(innerdata_sig_val)),
        "Nbkg_SR_train": int(len(innerdata_bkg_train) + len(innerdata_bkg_val)),
        "events_in_SR": len(innerdata_train) + len(innerdata_val),
        "labels": innerdata_test[:, -1],
        "preds": sup_preds_test,
    }
    with open(join(pkl_dir, "SUPresults.pkl"), "wb") as file:
        pkl.dump(results, file)


if args.skip_iad:
    print("Skipping IAD classifier.")
else:

    #FOR IAD CLASSIFIER
    # assigning label 1 to data (bkg+small signal) JUST FOR TRAINING
    clsf_innerdata_train = innerdata_train.copy()
    clsf_innerdata_train[:, -1] = 1

    clsf_innerdata_val = innerdata_val.copy()
    clsf_innerdata_val[:, -1] = 1

    # mixing together and shuffling
    iad_clsf_train_set = np.vstack([SR_extraBkg_IAD_train, clsf_innerdata_train])
    iad_clsf_val_set = np.vstack([SR_extraBkg_IAD_val, clsf_innerdata_val])

    iad_clsf_train_set = shuffle(iad_clsf_train_set, random_state=42)
    iad_clsf_val_set = shuffle(iad_clsf_val_set, random_state=42)

    #lets now train an IAD
    iad_scaler = StandardScaler()
    iad_scaler.fit(clsf_innerdata_train[:, 1:-1])

    iad_X_train = iad_scaler.transform(iad_clsf_train_set[:, 1:-1])
    iad_y_train = iad_clsf_train_set[:, -1]
    iad_X_val = iad_scaler.transform(iad_clsf_val_set[:, 1:-1])
    iad_y_val = iad_clsf_val_set[:, -1]

    # Let's protect ourselves from accidentally overwriting a trained model.
    if not exists(join(iad_classifier_savedir, "CLSF_val_losses.npy")):
        iad_classifier_model = make_classifier(save_path=iad_classifier_savedir,
                                               n_inputs=iad_X_train.shape[1],
                                               early_stopping=True, epochs=None,
                                               batch_size=args.batch_size,
                                               verbose=True)
        iad_classifier_model.fit(iad_X_train, iad_y_train, iad_X_val, iad_y_val)
    else:
        print(f"The model exists already in {iad_classifier_savedir}. Remove first if you want to overwrite.")

    iad_classifier_model = make_classifier(save_path=iad_classifier_savedir,
                                           n_inputs=iad_clsf_train_set[:, 1:-1].shape[1],
                                           load=True)

    # now let's evaluate the signal extraction performance on the same test set
    iad_X_test = iad_scaler.transform(innerdata_test[:, 1:-1])

    iad_preds_test = iad_classifier_model.predict(iad_X_test)

    results = {
        "mode": "IAD",
        "sigs": args.sigs,
        "Nsig": args.Nsig,
        "min": args.min,
        "max": args.max,
        "exBkg": args.exBkg,
        "Nsig_SR_train": int(len(innerdata_sig_train) + len(innerdata_sig_val)),
        "events_SR_train_IAD": int(len(iad_clsf_train_set) + len(iad_clsf_val_set)),
        "events_in_SR": len(innerdata_train) + len(innerdata_val),
        "labels": innerdata_test[:, -1],
        "preds": iad_preds_test,
    }
    with open(join(pkl_dir, "IADresults.pkl"), "wb") as file:
        pkl.dump(results, file)


if args.skip_cathode:
    print("Skipping CATHODE.")
    print("All trainings done!")
    raise SystemExit

#--------------------------------------------------------------------------------
# CATHODE: train a conditional flow (CFM) on the sidebands

m_scaler = StandardScaler() #only when in GeV
outer_scaler = make_pipeline(LogitScaler(), StandardScaler())

m_train = m_scaler.fit_transform(outerdata_train[:, 0:1])
m_val = m_scaler.transform(outerdata_val[:, 0:1])
X_train = outer_scaler.fit_transform(outerdata_train[:, 1:-1])
X_val = outer_scaler.transform(outerdata_val[:, 1:-1])

print(f"m_train{m_train.shape}")
print(f"X_train{X_train.shape}")
print(f"m_val{m_val.shape}")
print(f"X_Val{X_val.shape}")

#Replace NaNs with 0
X_train = np.nan_to_num(X_train, nan=0)
X_val = np.nan_to_num(X_val, nan=0)


def make_flow():
    # Same settings for training and loading, so the saved weights fit the model
    return ConditionalFlowMatching(
        save_path=flow_savedir,
        num_inputs=outerdata_train[:, 1:-1].shape[1],
        early_stopping=False,
        epochs=args.cfm_epochs,
        lr=1e-4,
        use_transformer=True,
        verbose=False,
    )


def get_latest_epoch(checkpoint_dir):
    """Find the latest saved epoch from DE_epoch_*.par files."""
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("DE_epoch_") and f.endswith(".par")]
    if not files:
        return None
    epochs = [int(f.replace("DE_epoch_", "").replace(".par", "")) for f in files]
    return max(epochs)


checkpoint_dir = join(flow_savedir, "DE_models")
latest_epoch = get_latest_epoch(checkpoint_dir) if exists(checkpoint_dir) else None

if args.skip_cfm_training or (latest_epoch is not None and latest_epoch >= args.cfm_epochs - 1):
    # Load the last saved epoch explicitly (does not depend on DE_val_losses.npy)
    if latest_epoch is None:
        raise FileNotFoundError(f"No trained CFM found in {checkpoint_dir}")
    print(f"Loading trained CFM from {flow_savedir}, epoch {latest_epoch}")
    flow_model = make_flow()
    flow_model.load_epoch_model(latest_epoch)

elif latest_epoch is None:
    print("No checkpoint found. Training from scratch...")
    flow_model = make_flow()
    flow_model.fit(X_train, m_train, X_val, m_val)

else:
    print(f"Found checkpoint at epoch {latest_epoch}. Resuming training...")
    flow_model = make_flow()

    # this loads the model + optimizer + scheduler
    flow_model.load_epoch_model(latest_epoch)

    print(f"Loaded checkpoint at epoch {latest_epoch}.")
    print(f"Remaining epochs: {args.cfm_epochs - latest_epoch - 1}")

    flow_model.fit(
        X_train, m_train, X_val, m_val,
        start_epoch=latest_epoch + 1
    )


# we also perform a logit first to stretch out the hard boundaries
SRm_scaler = LogitScaler(epsilon=1e-8)
SRm_train_KDE = SRm_scaler.fit_transform(innerdata_train[:, 0:1])

kde_model = KernelDensity(bandwidth=0.01, kernel='gaussian')
kde_model.fit(SRm_train_KDE)

# now just sample some ammount (one can also over sample to get more statistics)
SRm_samples = kde_model.sample(len(m_train)).astype(np.float32)
SRm_samples = SRm_scaler.inverse_transform(SRm_samples)

print(f"SRm_samples_shape:{SRm_samples.shape}")

_, binning, _ = plt.hist(innerdata_train[:, 0:1], bins=100, histtype="step", label="train", density=True)
plt.hist(SRm_samples, bins=binning, histtype="step", label="KDE", density=True)
plt.title("KDE in signal region")
plt.legend()
plt.savefig(join(pkl_dir, "KDE_SR.pdf"))
plt.close()


# drawing X samples from the flow model with the KDE samples (m) as conditional
#here we again use the stdscaler on m
SRm_samples = m_scaler.transform(SRm_samples)
SR_X_samples = flow_model.sample(n_samples=len(SRm_samples), m=SRm_samples)

SR_X_samples = outer_scaler.inverse_transform(SR_X_samples)
SRm_samples = m_scaler.inverse_transform(SRm_samples)

print(SR_X_samples.shape)

# assigning "signal" label 0 to samples - this is pure bkg
SR_samples = np.hstack([SRm_samples, SR_X_samples, np.zeros((SRm_samples.shape[0], 1))])

if args.save_samples:
    # mass + features, without the label column (used for the feature plots)
    np.save(join(pkl_dir, "CATHODE_SR_samples.npy"), SR_samples[:, :-1])
    print(f"Saved generated SR samples to {join(pkl_dir, 'CATHODE_SR_samples.npy')}")

# comparing samples to inner background (idealized sanity check)
print(f"innerdata_test shape{innerdata_test.shape}")
print(f"SR_samples shape{SR_samples.shape}")

for i in range(innerdata_test[:, :-1].shape[1]):
    # Upper panel (histograms)
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(7, 6))

    counts_data, binning , _ = ax[0].hist(innerdata_test[innerdata_test[:, -1] == 0, i],
                                        bins=100, label="Data Background",
                                        density=True, histtype="step")

    counts_sample, _, _ = ax[0].hist(SR_samples[:, i],
                                    bins=binning, label="Sampled Background",
                                    density=True, histtype="step")

    ax[0].legend(title="SR")
    ax[0].set_ylabel("Counts (norm.)")
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.2)

    # Ratio plot
    bin_centers = 0.5 * (binning[1:] + binning[:-1])
    ratio = np.divide(counts_sample, counts_data, out=np.zeros_like(counts_sample), where=counts_data != 0)

    ax[1].scatter(bin_centers, ratio, color='black', s=10)
    ax[1].axhline(1.0, color='red', linestyle='--')
    ax[1].set_ylabel("Ratio (Sample/Data)")
    ax[1].set_xlabel(branches[i])
    ax[1].set_ylim(0, 2)  # Adjust as needed for better visualization

    plt.tight_layout()
    plt.savefig(join(pkl_dir, f"{branches[i]}_SR.png"))
    plt.close()


samples_train = SR_samples[:len(SR_extraBkg_IAD_train)]
samples_val = SR_samples[len(SR_extraBkg_IAD_train):len(SR_extraBkg_IAD_train) + len(SR_extraBkg_IAD_val)]
# no need to separate SB and SR
print("CATHODE samples: training:", samples_train.shape)
print("CATHODE samples: validation:", samples_val.shape)

#FOR CATHODE CLASSIFIER

# assigning label 1 to data (bkg+small signal) JUST FOR TRAINING
clsf_innerdata_train = innerdata_train.copy()
clsf_innerdata_train[:, -1] = 1

clsf_innerdata_val = innerdata_val.copy()
clsf_innerdata_val[:, -1] = 1

# mixing together and shuffling
ca_clsf_train_set = np.vstack([samples_train, clsf_innerdata_train])
ca_clsf_val_set = np.vstack([samples_val, clsf_innerdata_val])

ca_clsf_train_set = shuffle(ca_clsf_train_set, random_state=42)
ca_clsf_val_set = shuffle(ca_clsf_val_set, random_state=42)

ca_scaler = StandardScaler()
ca_scaler.fit(clsf_innerdata_train[:, 1:-1])

ca_X_train = ca_scaler.transform(ca_clsf_train_set[:, 1:-1])
ca_y_train = ca_clsf_train_set[:, -1]
ca_X_val = ca_scaler.transform(ca_clsf_val_set[:, 1:-1])
ca_y_val = ca_clsf_val_set[:, -1]

# Let's protect ourselves from accidentally overwriting a trained model.
if not exists(join(ca_classifier_savedir, "CLSF_val_losses.npy")):
    ca_classifier_model = make_classifier(save_path=ca_classifier_savedir,
                                          n_inputs=ca_X_train.shape[1],
                                          early_stopping=True, epochs=None,
                                          batch_size=args.batch_size,
                                          verbose=True)
    ca_classifier_model.fit(ca_X_train, ca_y_train, ca_X_val, ca_y_val)
else:
    print(f"The model exists already in {ca_classifier_savedir}. Remove first if you want to overwrite.")

ca_classifier_model = make_classifier(save_path=ca_classifier_savedir,
                                      n_inputs=ca_clsf_train_set[:, 1:-1].shape[1],
                                      load=True)

# now let's evaluate the signal extraction performance on the same test set
ca_X_test = ca_scaler.transform(innerdata_test[:, 1:-1])
ca_preds_test = ca_classifier_model.predict(ca_X_test)

results = {
    "mode": "CATHODE",
    "sigs": args.sigs,
    "Nsig": args.Nsig,
    "min": args.min,
    "max": args.max,
    "exBkg": args.exBkg,
    "Nsig_SR_train": int(len(innerdata_sig_train) + len(innerdata_sig_val)),
    "events_SR_train_CATHODE": int(len(ca_clsf_train_set) + len(ca_clsf_val_set)),
    "events_in_SR": len(innerdata_train) + len(innerdata_val),
    "labels": innerdata_test[:, -1],
    "preds": ca_preds_test,
}
print(f"Saving CATHODE results to {pkl_dir}")
with open(join(pkl_dir, "CATHODEresults.pkl"), "wb") as file:
    pkl.dump(results, file)

print("All trainings done!")
