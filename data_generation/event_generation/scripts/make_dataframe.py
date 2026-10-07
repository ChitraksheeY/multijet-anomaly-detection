"""
Convert a Delphes ROOT file output to an LHCO-style h5 file.

Events are kept if they have at least one jet with pT > 1200 GeV and |eta| < 2.5
(the LHC Olympics 2020 selection). For each kept event, the 700 highest-pT
particle-flow candidates (tracks, photons, neutral hadrons) are stored as
(pT, eta, phi), zero-padded to 700, giving 2100 columns plus a final column
fixed to 1, as in the LHCO 2020 format.

Output: ./h5_files/<process>/run_<index>.h5, relative to the working directory.
"""

import os
import argparse

import pandas as pd
import numpy as np
import uproot

parser = argparse.ArgumentParser(description='Convert Delphes ROOT output to LHCO-style h5')
parser.add_argument('--process', default='test', type=str, help='Madgraph process name')
parser.add_argument('--root_file', type=str, help='Delphes ROOT file')
parser.add_argument('--index', default='0', type=str, help='Run label for output filename')
parser.add_argument('--lhe_file', type=str, help='LHE file name')
parser.add_argument('--label', default=0, type=int, help='Label in the last column: 0 = background, 1 = signal')
args = parser.parse_args()

process = args.process


event = uproot.open(args.root_file)["Delphes;1"]

event_fatjet = event['Jet']
gen_pt = np.asarray(event_fatjet['Jet.PT'])
gen_eta = np.asarray(event_fatjet['Jet.Eta'])

eflow_track = event['EFlowTrack']
eflow_photon = event['EFlowPhoton']
eflow_neutralhadron = event['EFlowNeutralHadron']

eflow_track_pt = np.asarray(eflow_track['EFlowTrack.PT'])
eflow_track_eta = np.asarray(eflow_track['EFlowTrack.Eta'])
eflow_track_phi = np.asarray(eflow_track['EFlowTrack.Phi'])

eflow_photon_pt = np.asarray(eflow_photon['EFlowPhoton.ET'])
eflow_photon_eta = np.asarray(eflow_photon['EFlowPhoton.Eta'])
eflow_photon_phi = np.asarray(eflow_photon['EFlowPhoton.Phi'])

eflow_neutralhadron_pt = np.asarray(eflow_neutralhadron['EFlowNeutralHadron.ET'])
eflow_neutralhadron_eta = np.asarray(eflow_neutralhadron['EFlowNeutralHadron.Eta'])
eflow_neutralhadron_phi = np.asarray(eflow_neutralhadron['EFlowNeutralHadron.Phi'])

allevents = []

for index, data in enumerate(gen_pt):

    # LHCO selection: at least one jet with pT > 1200 GeV and |eta| < 2.5
    Nj1200 = 0
    for j in range(len(data)):
        if (data[j] > 1200) & (np.abs(gen_eta[index][j]) < 2.5):
            Nj1200 += 1

    if Nj1200 >= 1:
        subjetlist = []
        for i in range(len(eflow_track_pt[index])):
            subjetlist.append([eflow_track_pt[index][i], eflow_track_eta[index][i], eflow_track_phi[index][i]])

        for i in range(len(eflow_photon_pt[index])):
            subjetlist.append([eflow_photon_pt[index][i], eflow_photon_eta[index][i], eflow_photon_phi[index][i]])

        for i in range(len(eflow_neutralhadron_pt[index])):
            subjetlist.append([eflow_neutralhadron_pt[index][i], eflow_neutralhadron_eta[index][i], eflow_neutralhadron_phi[index][i]])

        subjetlist = np.array(subjetlist)

        # keep the 700 highest-pT candidates, zero-pad the rest
        subjetlist = subjetlist[np.argsort(subjetlist[:, 0])][::-1]
        subjetlist = subjetlist[:700]
        subjetlist = np.pad(subjetlist, ((0, 700 - subjetlist.shape[0]), (0, 0)), 'constant', constant_values=0)

        allevents.append(subjetlist.reshape(700 * 3,))

df = pd.DataFrame(np.array(allevents))

# last column fixed to the specified label provided
df[2100] = args.label

os.makedirs(f'./h5_files/{process}', exist_ok=True)
df.to_hdf(f'./h5_files/{process}/run_{args.index}.h5', key='df', mode='w')

print(df.shape)

