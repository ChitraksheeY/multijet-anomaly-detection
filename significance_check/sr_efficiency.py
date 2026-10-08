import argparse
import subprocess
import tempfile
from os.path import join

import numpy as np
import uproot as up

parser = argparse.ArgumentParser(description="Make a shape file for Combine")
parser.add_argument("--background", type=str, help="Background sample")
parser.add_argument("--signal", type=str, help="Signal sample")
parser.add_argument("--branch", type=str, default="m_rsd", help="Branch to bin")
parser.add_argument("--min", type=float, default=3000., help="Min value")
parser.add_argument("--max", type=float, default=4000., help="Max value")
parser.add_argument("--nbkg", type=int, default=996800, help="Expected background yield")
parser.add_argument("--nsig", type=int, default=3200, help="Expected signal yield")
parser.add_argument("--signal2", type=str, default=None, help="Second signal to add")
parser.add_argument("--nsig2", type=int, default=0, help="Expected yield for signal 2")

args = parser.parse_args()

with up.open(args.background) as f:
    background = f["lhco_blackbox"][args.branch].array(library="np")

with up.open(args.signal) as f:
    signal = f["lhco_blackbox"][args.branch].array(library="np")

if args.signal2 is not None:
    with up.open(args.signal2) as f:
        signal2 = f["lhco_blackbox"][args.branch].array(library="np")

bkg_count = np.sum((background > args.min) & (background <= args.max)) * args.nbkg / len(background)
sig_count = np.sum((signal > args.min) & (signal <= args.max)) * args.nsig / len(signal)

if args.signal2 is not None:
    sig_count += np.sum((signal2 > args.min) & (signal2 <= args.max)) * args.nsig2 / len(signal2)

print(f'Total:   \t{args.nbkg}\t{args.nsig + args.nsig2}')
print(f'Selected:\t{int(bkg_count)}\t{int(sig_count)}')
print(f'Efficiency:\t{bkg_count / args.nbkg:.3f}\t{sig_count / (args.nsig + args.nsig2):.3f}')
print(f'Significance:\t{sig_count / bkg_count**0.5:2f}')
