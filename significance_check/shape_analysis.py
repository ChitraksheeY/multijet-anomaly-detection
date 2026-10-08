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
parser.add_argument("--min", type=float, default=2000., help="Min value")
parser.add_argument("--max", type=float, default=7000., help="Max value")
parser.add_argument("--bins", type=int, default=500, help="Bin count")
parser.add_argument("--nbkg", type=int, default=996800, help="Expected background yield")
parser.add_argument("--nsig", type=int, default=3200, help="Expected signal yield")
parser.add_argument("--signal2", type=str, default=None, help="Second signal to add")
parser.add_argument("--nsig2", type=int, default=0, help="Expected yield for signal 2")

args = parser.parse_args()

with up.open(args.background) as f:
    background = f["lhco_blackbox"][args.branch].array(library="np")

with up.open(args.signal) as f:
    signal = f["lhco_blackbox"][args.branch].array(library="np")

bins = np.linspace(args.min, args.max, args.bins + 1)

bkghist, _ = np.histogram(background, bins=bins)
bkg_norm = args.nbkg / len(background)
bkghist = bkghist * bkg_norm

sighist, _ = np.histogram(signal, bins=bins)
sig_norm = args.nsig / len(signal)
sighist = sighist * sig_norm

if args.signal2 is not None:
    with up.open(args.signal2) as f:
        signal2 = f["lhco_blackbox"][args.branch].array(library="np")
        sig2hist, _ = np.histogram(signal2, bins=bins)
        sig2_norm = args.nsig2 / len(signal2)
        sighist += sig2hist * sig2_norm

datahist = np.round(bkghist + sighist)

with tempfile.TemporaryDirectory() as tmpdirname:
    with open(join(tmpdirname, "shapes.csv"), "w", encoding="utf-8") as f:
        f.write("channel,process,systematic,bin,sum_w,sum_ww\n")
        for i, val in enumerate(bkghist):
            f.write(f"bin1,background,nominal,{i},{val},{val}\n")
        for i, val in enumerate(sighist):
            f.write(f"bin1,signal,nominal,{i},{val},{val}\n")
        for i, val in enumerate(datahist):
            f.write(f"bin1,data_obs,nominal,{i},{val},{val}\n")

    with open(join(tmpdirname, "card.txt"), "w", encoding="utf-8") as f:
        f.write(f"""
imax 1
jmax 1
kmax *
--------------------------------------------------------------------------------
shapes * * shapes.csv $CHANNEL:$PROCESS $CHANNEL:$PROCESS:$SYSTEMATIC
--------------------------------------------------------------------------------
bin         bin1
observation {sum(datahist)}
--------------------------------------------------------------------------------
bin             bin1   bin1
process         signal background
process         0      1
rate            {int(sum(sighist))}   {int(sum(bkghist))}
--------------------------------------------------------------------------------
lumi    lnN     1.10   1.0
bgnorm  lnN     1.00   1.3
""")

    subprocess.run(["text2workspace.py", "card.txt"], cwd=tmpdirname)
    subprocess.run(["combine", "-M", "Significance", "card.root"], cwd=tmpdirname)
