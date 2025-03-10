import glob
import os
import pandas as pd

if not os.path.exists('data/peaks'):
    os.mkdir('data/peaks')

main_dir = 'data/dap_download_may2016_peaks/dap_data_v4/peaks/'
summary_stats = []
for family in os.listdir(main_dir):
    files = glob.glob(f'{main_dir}/{family}' + '/**/*.narrowPeak', recursive=True)
    family_narrowPeaks = []
    for file in files:
        peaks = pd.read_csv(file, sep='\t', header=None)
        peaks[0] = peaks[0].str.replace('chr', '')
        peaks = peaks[[0, 1, 2]]
        peaks = peaks[peaks[1] >= 0]
        family_narrowPeaks.append(peaks)

    family_narrowPeaks = pd.concat(family_narrowPeaks)
    family_narrowPeaks.drop_duplicates(subset=[0, 1, 2], inplace=True)
    summary_stats.append([family.replace('_tnt', ''), family_narrowPeaks.shape[0]])
    family_narrowPeaks.to_csv(path_or_buf=f'data/peaks/{family}.bed', sep='\t', index=False, header=False)

pd.DataFrame(summary_stats, columns=['family', 'number of peaks']).to_csv(path_or_buf='data/summary_stats.csv',
                                                                          sep='\t', index=False)
