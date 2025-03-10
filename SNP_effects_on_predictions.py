import pandas as pd
import numpy as np
import pyranges as pr
from pybedtools import BedTool
from pyfaidx import Fasta
from utils import one_hot_encode
from tensorflow.keras.models import load_model
import seaborn as sns
pd.options.display.width=0
sns.set_context(context="paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
promoter_size = 1500
overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', nrows=2, index_col=0)
tf_families = overlap_mat.columns.tolist()
gwas_hits = pd.read_csv(filepath_or_buffer='data/gwas_hits/aragwas_associations_significant_permutation.csv')
gwas_hits.dropna(inplace=True, how='any', subset=['snp.position'])
gwas_hits['snp.position'] = gwas_hits['snp.position'].astype(int)
gwas_hits['chrom'] = gwas_hits['snp.chr'].str.replace('chr', '')
gwas_hits['start'] = gwas_hits['snp.position']
gwas_hits['end'] = gwas_hits['snp.position']
gwas_hits = gwas_hits[['chrom', 'start', 'end', 'snp.ref', 'snp.alt', 'maf', 'snp.annotations.0.effect',
                       'snp.annotations.0.impact', 'study.phenotype.name']]
print(gwas_hits.head())
print(f"Total number of GWAS hits {gwas_hits.shape[0]}")

onehot_dict = {'A': [1, 0, 0, 0],
                   'C': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1]}
nucl_to_onehot_position = {'A': 0, 'C':1, 'G': 2, 'T': 3}

def get_cis_coords(upstream=1000, downstream=500):
    gene_models = pr.read_gtf('data/annotation/Arabidopsis_thaliana.TAIR10.59.gtf', as_df=True)
    gene_models = gene_models[gene_models['Feature'] == 'gene']
    gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    gene_models = gene_models[gene_models['Chromosome'].isin(['1', '2', '3', '4', '5'])]

    cis_coords = []
    for chrom, start, end, strand, gene_id in gene_models.values:
        if strand == '+':
            cis_coords.append([chrom, max(0, start - upstream), start + downstream, gene_id, 'promoter'])
            cis_coords.append([chrom, max(0, end - downstream), end + upstream, gene_id, 'terminator'])
        else:
            cis_coords.append([chrom, max(0, end - downstream), end + upstream, gene_id, 'promoter'])
            cis_coords.append([chrom, max(0, start - upstream), start + downstream, gene_id, 'terminator'])
    cis_coords = pd.DataFrame(cis_coords)
    return cis_coords


def create_dataset(chrom, genome, promoter_coords):
    genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    promoter_coords[[0, 9]] = promoter_coords[[0, 9]].astype('str')
    promoter_coords = promoter_coords[promoter_coords[9] == chrom]

    seqs, snp_seqs, snp_ids, meta_info = [], [], [], []
    for coord in promoter_coords.values:
        chrom, start, end = coord[9], coord[10], coord[11]
        snp_ref, snp_alt, snp_pos, snp_pheno = coord[3], coord[4], coord[1]-1, coord[8]
        enc_seq = one_hot_encode(genome[chrom][start:end])
        enc_seq_snp = one_hot_encode(genome[chrom][start:end])
        enc_seq_snp[snp_pos - start, :] = onehot_dict[snp_alt]
        if enc_seq.shape[0] == promoter_size:
            seqs.append(enc_seq)
            snp_seqs.append(enc_seq_snp)
            snp_ids.append(f"{chrom}.{snp_pos+1}.{snp_ref}_{snp_alt}.{snp_pheno}")
            meta_info.append([coord[0], coord[1], coord[3], coord[4], coord[5],
                              coord[6], coord[8], coord[12], coord[13], coord[10], coord[11]])

    return np.array(seqs), np.array(snp_seqs), snp_ids, meta_info


prom_coords_df = get_cis_coords(upstream=1000, downstream=500)
prom_coords_df.to_csv(path_or_buf='data/promoter_coords.bed', index=False, header=False, sep='\t')

inters_results = BedTool.from_dataframe(gwas_hits).intersect('data/promoter_coords.bed', wa=True, wb=True)
inters_results = inters_results.to_dataframe(disable_auto_names=True, header=None)
inters_results.drop_duplicates(keep='first', inplace=True, subset=[0, 1])
print(inters_results.head())
print(f"Total number of GWAS hits intersecting promoters {inters_results.shape[0]}")

snp_effect_preds, snp_meta_infos = [], []
for row_idx, chrom_name in enumerate(['1', '2', '3', '4', '5']):
    orig_seq, mut_seq, snp_identifiers, meta_infos = create_dataset(chrom=chrom_name,
                                                                    genome='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa',
                                                                    promoter_coords=inters_results)
    model = load_model(f'saved_models/model_chrom_{chrom_name}_model.h5')
    predictions_original, predictions_mutated = [], []
    for idx in range(promoter_size // 250):
        preds = np.expand_dims(model.predict(orig_seq[:, 250 * idx:250 * (idx + 1), :]), axis=0)
        preds = preds > 0.5
        preds = preds.astype('int')
        predictions_original.append(preds)
    predictions_original = np.concatenate(predictions_original, axis=0)
    predictions_original = predictions_original.sum(axis=0)

    for idx in range(promoter_size // 250):
        preds = np.expand_dims(model.predict(mut_seq[:, 250 * idx:250 * (idx + 1), :]), axis=0)
        preds = preds > 0.5
        preds = preds.astype('int')
        predictions_mutated.append(preds)
    predictions_mutated = np.concatenate(predictions_mutated, axis=0)
    predictions_mutated = predictions_mutated.sum(axis=0)

    mutation_effect = predictions_original - predictions_mutated
    mutation_effect = pd.DataFrame(mutation_effect, columns=tf_families, index=snp_identifiers)
    snp_effect_preds.append(mutation_effect)
    snp_meta_infos.append(pd.DataFrame(meta_infos,
                                       columns=['Chrom', 'Pos', 'snp.ref', 'snp.alt', 'maf', 'snp.annotations.0.effect',
                                                'study.phenotype.name', 'gene_id', 'cis_region', 'start', 'end'],
                                       index=snp_identifiers))

snp_effect_preds = pd.concat(snp_effect_preds)
snp_meta_infos = pd.concat(snp_meta_infos)
print(snp_effect_preds.head())
snp_effect_preds.to_csv(path_or_buf='data/snp_effect_on_cis_regions.csv')
snp_meta_infos.to_csv(path_or_buf='data/snp_meta_infos.csv', index=False)
total = snp_effect_preds.shape[0]
subset_with_effects = len(np.where(snp_effect_preds.abs().sum(axis=1) != 0)[0])
print(f'Total: {total}, those with effects {subset_with_effects}, proportion {subset_with_effects/total}')

snp_effect_preds.reset_index(inplace=True)
snp_meta_infos.reset_index(inplace=True)
snp_effect_preds_and_meta_infos = snp_meta_infos.merge(snp_effect_preds, how='inner', on='index')
snp_effect_preds_and_meta_infos.rename(columns={'index':'snp_ID'}, inplace=True)
snp_effect_preds_and_meta_infos.to_csv(path_or_buf='data/snp_effect_preds_and_meta_infos.csv', index=False)
print(snp_effect_preds_and_meta_infos.head())
snp_effect_preds_and_meta_infos.to_csv(path_or_buf='results/Supplementary_table_2_sheet_1.csv', index=False)
