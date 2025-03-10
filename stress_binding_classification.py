import pandas as pd
import numpy as np
from utils import one_hot_encode
from pyfaidx import Fasta
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_score, recall_score
import seaborn as sns
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
sns.set_context(context="paper", rc={"font.size":40,"axes.titlesize":14,"axes.labelsize":14})

pd.options.display.width=0
np.random.seed(42)

overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', index_col=0, nrows=3)
tfs = [i.replace('_tnt', '') for i in overlap_mat.columns.tolist()]
root_data_dir = 'data/heat_stress_data_mays'
all_peaks = pd.read_excel(io=f"{root_data_dir}/stress_related_peaks.ods",
                          usecols=['chrom', 'start', 'end', 'Conc_Control', 'Conc_Heat', 'Fold'],
                          dtype={'chrom': str})
all_peaks = all_peaks[all_peaks["chrom"].isin([str(i) for i in range(1, 11)])]

# Plot FC from stress recorded peaks
fig_fc, ax_fc = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
_, _, bars = ax_fc.hist(all_peaks['Fold'].values, bins=np.arange(-2.5, 2.5, 0.1), edgecolor='white')
for bar in bars:
    if -1 <= bar.get_x() <= 1:
        bar.set_facecolor('#E6C767')
    else:
        bar.set_facecolor('#898121')
ax_fc.set_ylabel('peaks')
ax_fc.set_xlabel('fold')
fig_fc.tight_layout()
plt.savefig(f"results/Figures/stress_FC_distribution.svg", bbox_inches='tight',
                dpi=300, format='svg')
plt.show()

for subset, c in zip(['extreme', 'less_extreme', 'all'], ['#898121', '#E6C767', 'white']):
    peaks = all_peaks.copy()
    if subset == 'extreme':
        peaks = peaks[peaks['Fold'].abs() >= 1.0]
    elif subset == 'less_extreme':
        peaks = peaks[peaks['Fold'].abs() < 1.0]
    print(peaks.head())
    print(peaks.shape)
    peaks['target'] = [0 if i < 0 else 1 for i in peaks['Fold']]
    print(peaks.head())
    print(peaks.shape)
    
    genome = Fasta(filename=f"{root_data_dir}/Zea_mays.B73_RefGen_v4.dna.toplevel.fa", as_raw=True, read_ahead=10000,
                   sequence_always_upper=True)
    seqs, targets, folds, chroms = [], [], [], []
    for chrom, start, end, _, _, fold, target in peaks.values:
        mid_point = (start + end) // 2
        seq = one_hot_encode(genome[chrom][mid_point-125:mid_point+125])
        if seq.shape[0] == 250:
            seqs.append(seq)
            targets.append(target)
            folds.append(fold)
            chroms.append(chrom)
    seqs, targets, folds, chroms = np.array(seqs), np.array(targets), np.array(folds), np.array(chroms)
    
    means_importance, feature_names = [], []
    results = []
    
    for ara_chrom in ['1', '2', '3', '4', '5']:
        dap_seq_model = load_model(f'saved_models/model_chrom_{ara_chrom}_model.h5')
        predictions = dap_seq_model.predict(seqs)
        unique_ls_of_chroms = np.unique(chroms)
        for train_chroms_idx, val_chroms_idx in KFold(n_splits=5).split(unique_ls_of_chroms):
            train_chroms = unique_ls_of_chroms[train_chroms_idx]
            val_chroms = unique_ls_of_chroms[val_chroms_idx]
            train_x_idx = np.nonzero(np.isin(chroms, train_chroms))[0]
            val_x_idx = np.nonzero(np.isin(chroms, val_chroms))[0]
            x_train, y_train = predictions[train_x_idx], targets[train_x_idx]
            x_val, y_val = predictions[val_x_idx], targets[val_x_idx]
    
            # Balance x_train since one class is much larger than the other
            class_0, class_1 = np.where(y_train == 0)[0], np.where(y_train == 1)[0]
            num_to_select = min(len(class_0), len(class_1))
            sel_indices_0 = np.random.choice(class_0, size=num_to_select, replace=False)
            sel_indices_1 = np.random.choice(class_1, size=num_to_select, replace=False)
            x = np.concatenate([
                np.take(a=x_train, indices=sel_indices_0, axis=0),
                np.take(a=x_train, indices=sel_indices_1, axis=0)
            ])
            y = np.concatenate([
                np.take(a=y_train, indices=sel_indices_0, axis=0),
                np.take(a=y_train, indices=sel_indices_1, axis=0)
            ])
            y_shuffled = y.copy()
            np.random.shuffle(y_shuffled)
    
            # Fit logistic regression
            classifier = LogisticRegression(penalty='l1', solver='liblinear')
            classifier.fit(x, y)
    
            # Fit logistic regression on data with shuffled labels
            classifier_shuffled = LogisticRegression(penalty='l1', solver='liblinear')
            classifier_shuffled.fit(x, y_shuffled)
    
            # Predict on validation data. Since it is also not balance, we will compute multiple performance metrics for it
            pred_cls = classifier.predict(x_val)
            pred_shuffled_cls = classifier_shuffled.predict(x_val)
    
            pred_probs = classifier.predict_proba(x_val)[:, 1]
            pred_shuffled_probs = classifier_shuffled.predict_proba(x_val)[:, 1]
    
            acc = balanced_accuracy_score(y_val, pred_cls)
            acc_shuffled = balanced_accuracy_score(y_val, pred_shuffled_cls)
    
            precision = precision_score(y_val, pred_cls, average='weighted')
            precision_shuffled = precision_score(y_val, pred_shuffled_cls, average='weighted')
    
            recall = recall_score(y_val, pred_cls, average='weighted')
            recall_shuffled = recall_score(y_val, pred_shuffled_cls, average='weighted')
    
            auc = roc_auc_score(y_val, pred_probs)
            auc_shuffled = roc_auc_score(y_val, pred_shuffled_probs)
            results.append([acc, recall, precision, auc, 'model'])
            results.append([acc_shuffled, recall_shuffled, precision_shuffled, auc_shuffled, 'base'])
    
            print(f'Results for DAP-seq model that used chromosome {ara_chrom} as validation chromosome')
            print('------------------------------------------')
            print(f'Binary accuracy: control={acc_shuffled} test={acc}')
            print(f"Recall: control={recall_shuffled} test={recall}")
            print(f"Precision: control={precision_shuffled} test={precision}")
            print(f"AUC: control={auc_shuffled} test={auc}")
    
            # Compute permutation importance scores
            r = permutation_importance(classifier, x_val, y_val, n_repeats=30, random_state=42, scoring='balanced_accuracy')
            means_importance.extend(r.importances_mean)
            feature_names.extend(tfs)
    
    df_importance = pd.DataFrame(data={
        'mean importance': means_importance,
        'tf family': feature_names,
    })
    
    df_results = pd.DataFrame(data=results, columns=['accuracy', 'recall', 'precision', 'auc', 'fit'])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 10))
    order = df_importance.groupby(["tf family"])["mean importance"].mean().abs().sort_values(ascending=False).index
    
    sns.boxplot(data=df_importance, ax=ax, y='tf family', x='mean importance', order=order, whis=(0, 100),
                color=c, whiskerprops={'linestyle':'--'})
    plt.axvline(x=0, color='grey', linestyle='--')
    fig.tight_layout()
    plt.savefig(f"results/Figures/log_reg_importance_score_stress_{subset}.svg", bbox_inches='tight',
                dpi=300, format='svg')
    plt.show()

    fig_perf, ax_perf = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    for idx, metric in enumerate(['accuracy', 'auc']):
        sns.boxplot(data=df_results[[metric, 'fit']], x='fit', y=metric, ax=ax_perf[idx],
                    whiskerprops={'linestyle':'--'}, whis=(0, 100), order=['model', 'base'], color=c)
        ax_perf[idx].spines[['right', 'top']].set_visible(False)
        ax_perf[idx].set_ylim(0.3, 1)
        ax_perf[idx].axhline(y=0.5, color='silver', linestyle='--')
    
    fig_perf.tight_layout()
    plt.savefig(f"results/Figures/log_reg_performance_stress_{subset}.svg", bbox_inches='tight',
                dpi=300, format='svg')
    plt.show()
