import pandas as pd
from pyfaidx import Fasta
from utils import one_hot_encode
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
from tensorflow.keras.backend import clear_session
from mlcm import mlcm
import os

if not os.path.exists('results/Figures'):
    os.makedirs('results/Figures')


def create_dataset(genome, overlap_mat, coords):
    genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    x = []
    y = []

    for coord in coords:
        chrom, start, end = coord.split(':')
        seq = one_hot_encode(genome[chrom][int(start):int(end)])
        if seq.shape[0] == 250:
            x.append(seq)
            y.append(overlap_mat.loc[coord, :].tolist())

    return np.array(x), np.array(y)


preds, actual, labels = [], [], []
for chrom_name in ['1', '2', '3', '4', '5']:
    overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', index_col=0)
    overlap_mat['chrom'] = [i.split(':')[0] for i in overlap_mat.index.tolist()]
    overlap_mat = overlap_mat[overlap_mat['chrom'] == chrom_name]
    overlap_mat.drop(['chrom'], axis=1, inplace=True)
    labels.append(overlap_mat.columns.tolist())

    x, y = create_dataset(genome='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa',
                          overlap_mat=overlap_mat,
                          coords=overlap_mat.index.tolist())
    clear_session()
    model = models.load_model(f"saved_models/model_chrom_{chrom_name}_model.h5")
    y_preds = model.predict(x) > 0.5
    y_preds = y_preds.astype("int")
    print(f"Predictions for chrom: {chrom_name}")
    print(y.shape, y_preds.shape)
    preds.append(y_preds)
    actual.append(y)

preds, actual = np.concatenate(preds, axis=0), np.concatenate(actual, axis=0)
cm, norm_cm = mlcm.cm(actual, preds)
print(norm_cm.shape)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 20),
                       gridspec_kw={'height_ratios': [1, 5], 'width_ratios': [20, 4, 1]})
ax[1, 0].sharex(ax[0, 0])
ax[1, 0].sharey(ax[1, 1])
sns.heatmap(data=norm_cm[:-1, :-1], ax=ax[1, 0], xticklabels=labels[0], yticklabels=labels[0],
            cbar_ax=ax[1, 2], cmap='Purples')
x_tick_pos = [i + 0.5 for i in range(len(labels[0]))]
ax[0, 0].bar(x_tick_pos, preds.sum(axis=0), align='center', color='#371F76')
ax[1, 1].barh(x_tick_pos, actual.sum(axis=0), align='center', color='#371F76')
ax[0, 0].set_xticks(x_tick_pos)
ax[0, 0].set_xticklabels(labels[0])
ax[0, 1].axis('off')
ax[0, 2].axis('off')
plt.savefig(f"results/Figures/confusion_matrix_multilabel.svg", bbox_inches='tight', dpi=300, format='svg')
plt.show()

cm_skl = multilabel_confusion_matrix(actual, preds)
cm_skl = cm_skl.sum(axis=0)
cm_skl = cm_skl/cm_skl.sum(axis=0)
fig2, ax2 = plt.subplots(figsize=(4, 4))
sns.heatmap(cm_skl, annot=True, fmt='.2f', ax=ax2, linewidth=.5, cmap='Purples')
ax2.set(xlabel="Predicted", ylabel="True")
plt.savefig(f"results/Figures/confusion_matrix_compressed.svg", bbox_inches='tight', dpi=300, format='svg')
plt.show()

fig3, ax3 = plt.subplots(nrows=2, ncols=3, figsize=(20, 20),
                         gridspec_kw={'height_ratios': [1, 5], 'width_ratios': [20, 4, 1]})
ax3[1, 0].sharex(ax3[0, 0])
ax3[1, 0].sharey(ax3[1, 1])
masked_cm = norm_cm[:-1, :-1]
masked_cm[np.diag_indices_from(masked_cm)] = 0
cm = cm[:-1, :-1]
cm[np.diag_indices_from(cm)] = 0
pd.DataFrame(masked_cm, columns=labels[0], index=labels[0]).to_csv('data/diagonal_masked_cm.csv')
sns.heatmap(data=masked_cm, ax=ax3[1, 0], xticklabels=labels[0], yticklabels=labels[0],
            cbar_ax=ax3[1, 2], cmap='Reds')
x_tick_pos = [i + 0.5 for i in range(len(labels[0]))]
ax3[0, 0].bar(x_tick_pos, cm.sum(axis=0), align='center', color='#8B0000')
ax3[1, 1].barh(x_tick_pos, actual.sum(axis=0), align='center', color='#8B0000')
ax3[0, 0].set_xticks(x_tick_pos)
ax3[0, 0].set_xticklabels(labels[0])
ax3[0, 1].axis('off')
ax3[0, 2].axis('off')
plt.savefig(f"results/Figures/confusion_matrix_multilabel_masked_diag.svg", bbox_inches='tight',
            dpi=300, format='svg')
plt.show()
