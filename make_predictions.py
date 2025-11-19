import numpy as np
import pandas as pd
from Bio import SeqIO
import os
from utils import one_hot_encode
from tensorflow.keras.models import load_model
pd.options.display.width=0

tf_fam = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', nrows=3, index_col=0).columns.tolist()
tf_fam = [x.replace('_tnt', '') for x in tf_fam]

for f in os.listdir('data'):
    if f.endswith('.fas'):
        seqs_enc, seq_ids = [], []
        for rec in SeqIO.parse(handle=f"data/{f}", format='fasta'):
            if len(rec.seq) == 1501:
                seqs_enc.append(one_hot_encode(str(rec.seq)))
                seq_ids.append(rec.id)
        seqs_enc = np.array(seqs_enc)
        seq_ids = np.array(seq_ids)
        print(seqs_enc.shape, seq_ids.shape)

        ## I applied the Arabidopsis model that used chrom 1 as validation chromosome
        model = load_model(f'saved_models/model_chrom_1_model.h5')
        predictions = []
        for idx in range(seqs_enc.shape[1] // 250):
            preds = np.expand_dims(model.predict(seqs_enc[:, 250 * idx:250 * (idx + 1), :]), axis=0)
            preds = preds > 0.5
            preds = preds.astype('int')
            predictions.append(preds)

        predictions = np.concatenate(predictions, axis=0)
        predictions = predictions.sum(axis=0)
        predictions = pd.DataFrame(predictions, index=seq_ids, columns=tf_fam)
        print(predictions.shape, seq_ids.shape)
        print(predictions.head())
        predictions.to_csv(f"results/predictions_{f.split('.fas')[0]}.csv")

