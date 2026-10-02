"""Make a .npy dataset from the feature ROOT files.

Examples:
    python make_datasets.py events_BB3_KKg2gr_r500.root events_BB3_KKg2gr_r500.allfeatures.root sig_dijet.npy --preset event_level_Mrsd
    python make_datasets.py base.root features.root custom.npy --features m_rsd mj1 mj2 tau21min tau21max
"""

import argparse

import numpy as np
import ROOT

# Example feature sets (first column = resonant variable)
PRESETS = {
    "event_level_Mrsd": ["m_rsd", "HT", "tau1", "tau2", "tau3", "tau4"],
    "dijet_features":   ["mj1j2", "mjmin", "delta_mjj", "tau21max", "tau21min"],
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("filename_base", help="input ROOT file (output of h5_to_root.py)")
parser.add_argument("filename_derived", help="feature ROOT file (output of compute_features.run)")
parser.add_argument("output", help="output .npy file")
parser.add_argument("--preset", choices=PRESETS, default="event_level_Mrsd", help="example feature set")
parser.add_argument("--features", nargs="+", help="any list of branch names (overrides --preset)")
parser.add_argument("--seed", type=int, default=42, help="seed for shuffling")
args = parser.parse_args()

features = args.features if args.features else PRESETS[args.preset]

chain_base = ROOT.TChain("lhco_blackbox")
chain_base.Add(args.filename_base)
chain_derived = ROOT.TChain("lhco_blackbox")
chain_derived.Add(args.filename_derived)
chain_base.AddFriend(chain_derived)

df = ROOT.RDataFrame(chain_base)
df_sel = df.Filter("type == 1")    # all generated samples have type == 1

columns = df_sel.AsNumpy(features)
arr = np.stack([columns[name].astype(float) for name in features], axis=-1)
print(arr.shape)

rng = np.random.default_rng(args.seed)
np.save(args.output, arr[rng.permutation(arr.shape[0])])
print("Saved", args.output, "with columns:", ", ".join(features))
