import numpy as np
import os
import pandas as pd
import h5py


if not os.path.exists('results/Figures'):
    os.mkdir('results/Figures')


def get_predictive_pwms(mod_file, family_name):
    f = h5py.File(name=f'data/modisco/{family_name}/{mod_file}', mode="r")
    metacluster = f["metacluster_idx_to_submetacluster_results"]['metacluster_0']
    patterns = metacluster['seqlets_to_patterns_result']['patterns']
    motifs = []
    for pattern_idx, pattern_name in enumerate(patterns['all_pattern_names'][:1]):
        pattern = patterns[pattern_name.decode()]
        motif = pattern["sequence"]["fwd"][:]
        motifs.append(motif)

    return motifs[0]


for family in ['bHLH_tnt', 'bZIP_tnt', 'WRKY_tnt']:
    pred_motifs, tfs = [], []
    for file_name in os.listdir(f'data/modisco/{family}'):
        if file_name.endswith('modisco.hdf5'):
            pred_motif = get_predictive_pwms(mod_file=file_name, family_name=family)
            pred_motifs.append(pred_motif)
            tfs.append(file_name.split('_modisco')[0])
    print(len(pred_motifs), len(tfs), family)
    pred_motifs = np.array(pred_motifs)
    tfs = np.array(tfs)
    pd.Series(tfs).to_csv(path_or_buf=f'data/{family}_tfs.csv', index=False)
    os.system(f'rm -rf data/{family}_predictive_motifs.h5')
    h = h5py.File(name=f'data/{family}_predictive_motifs.h5', mode='w')
    h.create_dataset(name='motifs', data=pred_motifs)
    h.close()