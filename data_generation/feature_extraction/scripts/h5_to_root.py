import os
import glob
import argparse
import pandas as pd
import numpy as np
import ROOT

parser = argparse.ArgumentParser(description="Stitch run_*.h5 files into one ROOT file")
parser.add_argument("input_dir", help="directory with run_*.h5 files")
parser.add_argument("output_root", help="output ROOT file")
parser.add_argument("--tree_name", default="lhco_blackbox")
args = parser.parse_args()

input_dir = args.input_dir
output_root = args.output_root

files = sorted(glob.glob(os.path.join(input_dir, "run_*.h5")))
tree_name = args.tree_name

root_file = ROOT.TFile(output_root, "recreate")
tree = ROOT.TTree(tree_name, tree_name)

leaf = np.zeros((2100,), dtype=np.float64)
leaf_type = np.zeros((1,), dtype=np.int32)

tree.Branch("data", leaf, "data[2100]/D")
tree.Branch("type", leaf_type, "type/I")

count = 0
total_rows = 0
total_cols = None


def fill_tree(series):
    global count
    leaf[:] = series.values[:-1]
    leaf_type[0] = int(series.values[-1])
    tree.Fill()
    count += 1

for file in files:
    print(f"Processing {file}")

    try:
        df = pd.read_hdf(file)
    except Exception as e:
        print(f"Could not open file: {file}")
        print(f"Error: {e}")
        continue

    if total_cols is None:
        total_cols = df.shape[1]
    total_rows += df.shape[0]
    df.apply(fill_tree, axis=1)

tree.AutoSave()
root_file.Close()

print(f"Total events written: {count}")
