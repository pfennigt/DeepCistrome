import pyranges as pr
import pandas as pd
from pyfaidx import Fasta
from utils import one_hot_encode
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
import seaborn as sns
import os
from numba import njit
from sklearn.preprocessing import StandardScaler
import random
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sns.set_style("ticks")
sns.set_context(context="paper", rc={"font.size":40,"axes.titlesize":14,"axes.labelsize":14})
pd.options.display.width = 0


@njit
def set_seed():
    np.random.seed(42)
    random.seed(42)


set_seed()
gene_models = pr.read_gtf('data/annotation/Arabidopsis_thaliana.TAIR10.59.gtf', as_df=True)
gene_models = gene_models[gene_models['Feature'] == 'gene']
gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
print(gene_models.head())

overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', nrows=2, index_col=0)
tf_families = overlap_mat.columns.tolist()
tf_families = [x.replace('_tnt', '') for x in tf_families]

def create_dataset(df_gene_models, chrom, genome, upstream=1000, downstream=500):
    genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)

    seqs, gene_ids = [], []
    df_gene_models = df_gene_models[df_gene_models['Chromosome'] == chrom]
    for chrom, start, end, strand, gene_id in df_gene_models.values:
        if strand == '+':
            prom_start, prom_end = max(0, start - upstream), start + downstream
            term_start, term_end = max(0, end - downstream), end + upstream
        else:
            prom_start, prom_end = max(0, end - downstream), end + upstream
            term_start, term_end = max(0, start - upstream), start + downstream

        seq = np.concatenate([
            one_hot_encode(genome[chrom][prom_start:prom_end]),
            one_hot_encode(genome[chrom][term_start:term_end])
        ], axis=0)

        if seq.shape[0] == 2*(upstream+downstream):
            seqs.append(seq)
            gene_ids.append(gene_id)

    return np.array(seqs), np.array(gene_ids)


genome_file = 'data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa'
upstream_size, downstream_size = 1000, 500
prom_terms = []
genes_all = []
for chrom_name in ['1', '2', '3', '4', '5']:
    enc_seqs, genes = create_dataset(gene_models, chrom=chrom_name, genome=genome_file,
                                     upstream=upstream_size, downstream=downstream_size)
    print(enc_seqs.shape, genes.shape)

    clear_session()
    model = load_model(f'saved_models/model_chrom_{chrom_name}_model.h5')
    seq_len = 2*(upstream_size+downstream_size)
    predictions = []
    for idx in range(seq_len // 250):
        preds = np.expand_dims(model.predict(enc_seqs[:, 250 * idx:250 * (idx + 1), :]), axis=0)
        preds = preds > 0.5
        preds = preds.astype('int')
        predictions.append(preds)

    predictions = np.concatenate(predictions, axis=0)
    predictions = predictions.sum(axis=0)
    prom_terms.append(predictions)
    genes_all.append(genes)

prom_terms = np.concatenate(prom_terms, axis=0)
genes_all = np.concatenate(genes_all, axis=0)
data_to_save = pd.DataFrame(prom_terms, index=genes_all, columns=tf_families)

# Perform UMAP, cluster embeddings to get clusters
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 8))
u_mapper = UMAP(n_components=2, n_neighbors=30, random_state=42, min_dist=0.15, n_jobs=1,
                transform_seed=42,)
prom_std = StandardScaler().fit_transform(prom_terms)
embeddings = u_mapper.fit_transform(prom_std)
clusters = HDBSCAN(min_samples=30, min_cluster_size=400,
                   algorithm='kdtree', n_jobs=1).fit_predict(embeddings) + 1
print(f'Number of clusters: {np.unique(clusters)}')
colors = ['silver', '#be398d', '#c595fa', '#1a2421', '#ff7926', '#008080', '#dc143c', '#c154c1',
          '#79baec', '#202e51', '#7B3F00', '#be5504', '#0000ff', '#fc6c85', '#e6b400', '#ffefd5', '#7fffd4',
          '#bdb76b', '#dda0dd', '#2f4f4f', '#f5c26b', '#cae7d3', '#98f5ff', '#cd3333']
predicted_cluster_cols = [colors[x] for x in clusters]

data_to_save['cluster'] = clusters
data_to_save['cluster_col'] = predicted_cluster_cols
print(data_to_save.head())
data_to_save.to_csv(path_or_buf='data/prom_term_predictions.csv')

ax.scatter(embeddings[:, 0], embeddings[:, 1], c=predicted_cluster_cols, s=4)
ax.set_xlabel(f'Dim 1')
ax.set_ylabel(f'Dim 2')
ax.xaxis.grid(True, alpha=0.5)
ax.yaxis.grid(True, alpha=0.5)
plt.savefig(f"results/Figures/umap_prom_terms.svg", bbox_inches='tight', dpi=300, format='svg')
plt.savefig(f"results/Figures/umap_prom_terms.png", bbox_inches='tight', dpi=300, format='png')
plt.show()

cluster_to_probs = []
cluster_ids = []
cluster_ids_dict = {}
cluster_ids_to_df = []
for cluster_idx in np.unique(clusters):
    pred_copy = prom_terms.copy()
    pred_copy = pred_copy[np.where(clusters == cluster_idx)[0]]
    print(f'Cluster size: {pred_copy.shape[0]}')
    cluster_props = np.sum(pred_copy, axis=0)/np.sum(np.sum(pred_copy, axis=0))
    cluster_to_probs.append(cluster_props)

    # cluster naming
    if cluster_idx == 0:
        cluster_ids.append('-')
        cluster_ids_dict[cluster_idx] = '-'
        cluster_ids_to_df.append([cluster_idx, '-'])
    else:
        cluster_props_df = pd.DataFrame(data={'tf_families':tf_families, 'proportions':cluster_props})
        cluster_props_df.sort_values(by=['proportions'], inplace=True, ascending=False)
        cluster_props_df = cluster_props_df[~cluster_props_df['tf_families'].isin(['AP2EREBP', 'MYB', 'MYBrelated',
                                                                                   'C2C2dof', 'NAC', 'HB',
                                                                                   'G2like'])]
        cluster_name = f'c{cluster_idx}-'+'-'.join(cluster_props_df['tf_families'].tolist()[:4])
        cluster_ids.append(cluster_name)
        cluster_ids_dict[cluster_idx] = cluster_name
        cluster_ids_to_df.append([cluster_idx, cluster_name])

cluster_to_probs = pd.DataFrame(cluster_to_probs, columns=tf_families, index=cluster_ids)
print(cluster_to_probs.head())
cluster_to_probs.to_csv(path_or_buf='results/cluster_to_probs.csv')
row_cols = pd.Series(np.unique(clusters))
row_cols = row_cols.map({k: colors[k] for k in np.unique(clusters)})
row_cols.rename(index=cluster_ids_dict, inplace=True)
g = sns.clustermap(data=cluster_to_probs, cmap='Reds', linewidths=0.5, figsize=(15, 5),
                   cbar_pos=(.02, .06, .2, .04),
                   cbar_kws={"orientation": "horizontal", "pad": 0.01},
                   row_colors=row_cols, yticklabels=True, col_colors=None,
                   col_cluster=False)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
#g.ax_heatmap.tick_params(right=True)
plt.tight_layout()
plt.savefig(f"results/Figures/umap_prom_terms_heatmap.svg", bbox_inches='tight', dpi=300, format='svg')
plt.show()

cluster_ids_to_df = pd.DataFrame(cluster_ids_to_df, columns=['cluster_id', 'cluster_name'])
print(cluster_ids_to_df.head())
cluster_ids_to_df.to_csv('data/cluster_ids_cluster_name.csv', index=False)