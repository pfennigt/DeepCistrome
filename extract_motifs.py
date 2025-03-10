import h5py
import numpy as np
import os
import pandas as pd

if not os.path.exists('results/Figures'):
    os.mkdir('results/Figures')


def get_predictive_pwms(mod_file):
    f = h5py.File(name=f'data/modisco/{mod_file}', mode="r")
    metacluster = f["metacluster_idx_to_submetacluster_results"]['metacluster_0']
    patterns = metacluster['seqlets_to_patterns_result']['patterns']
    motifs = []
    for pattern_idx, pattern_name in enumerate(patterns['all_pattern_names'][:1]):
        pattern = patterns[pattern_name.decode()]
        motif = pattern["sequence"]["fwd"][:]
        motifs.append(motif)

    return motifs[0]


pred_motifs, tfs = [], []
for file_name in os.listdir('data/modisco'):
    if file_name.endswith('modisco.hdf5'):
        pred_motif = get_predictive_pwms(mod_file=file_name)
        pred_motifs.append(pred_motif)
        tfs.append(file_name.split('_modisco')[0])
print(len(pred_motifs), len(tfs))
pred_motifs = np.array(pred_motifs)
tfs = np.array(tfs)
pd.Series(tfs).to_csv(path_or_buf='data/tfs.csv', index=False)
os.system(f'rm -rf data/predictive_motifs.h5')
h = h5py.File(name='data/predictive_motifs.h5', mode='w')
h.create_dataset(name='motifs', data=pred_motifs)
h.close()

