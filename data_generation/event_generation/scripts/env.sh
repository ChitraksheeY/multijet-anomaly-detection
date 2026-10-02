# Environment for event generation. Edit the two paths below for your setup.

# Pythia 8.219 install prefix (needed to build DelphesPythia8)
export PYTHIA8=/path/to/pythia8219/install

# Delphes 3.4.1 build directory (contains DelphesPythia8)
export DELPHES_DIR=/path/to/delphes-3.4.1

export LD_LIBRARY_PATH=$PYTHIA8/lib:$LD_LIBRARY_PATH

# Python environment for make_dataframe.py (uproot, pandas, PyTables)
. /cvmfs/sft.cern.ch/lcg/views/LCG_102b/x86_64-centos9-gcc11-opt/setup.sh
