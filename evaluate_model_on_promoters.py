import pandas as pd
import pyranges as pr
from pybedtools import BedTool, Interval
from utils import InputGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

gtf_file = 'data/annotation/Arabidopsis_thaliana.TAIR10.59.gtf'
overlap_mat_path = 'data/overlap_matrix.bed'
overlap_mat = pd.read_csv(overlap_mat_path, sep='\t', index_col=0)


def generate_prom_bed(gtf):
    gtf_df = pr.read_gtf(gtf, as_df=True)
    gtf_df = gtf_df[gtf_df['Feature'] == 'gene']
    gtf_df = gtf_df[['Chromosome', 'Start', 'End', 'Strand']]
    bed = []
    for chrom, start, end, strand in gtf_df.values:
        if strand == '+':
            bed.append(Interval(str(chrom), int(max(0, start - 1000)), int(start+500)))
        else:
            bed.append(Interval(str(chrom), int(max(0, end - 500)), int(end+1000)))
    return BedTool(bed)


def generate_overlap_bed(mat):
    bed = []
    for coord in mat.index.tolist():
        chrom, start, end = coord.split(':')
        bed.append(Interval(str(chrom), int(start), int(end)))
    return BedTool(bed)


prom_bed = generate_prom_bed(gtf_file)
overlap_bed = generate_overlap_bed(overlap_mat)
prom_overlap = overlap_bed.intersect(prom_bed, u=True, f=0.3).to_dataframe()

prom_evals = []
for chrom in [1, 2, 3, 4, 5]:
    prom_overlap_chrom = prom_overlap.copy()
    prom_overlap_chrom = prom_overlap_chrom[prom_overlap_chrom['chrom'] == chrom]
    idx_ls = [f"{i[0]}:{i[1]}:{i[2]}" for i in prom_overlap_chrom.values]
    
    data_gen = InputGenerator(genome='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa',
                              bins_list=idx_ls, batch_size=512, overlap_mat=overlap_mat,
                              window_size=250, control='model')

    for model_name in ['model', 'di', 'si']:
        if model_name == 'model':
            model_type = 'promoters'
        else:
            model_type = model_name
        model = load_model(f'saved_models/model_chrom_{chrom}_{model_name}.h5')
        results = model.evaluate(data_gen)
        prom_evals.append([results[1], model_type, 'auROC'])
        prom_evals.append([results[2], model_type, 'auPR'])
        prom_evals.append([results[3], model_type, 'weighted auPR'])

pd.DataFrame(prom_evals, columns=['score', 'model type', 'metric']).to_csv(path_or_buf='results/prom_evals.csv',
                                                                           index=False)





