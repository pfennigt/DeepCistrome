import pandas as pd
import numpy as np
from tensorflow.keras import models
import shap
from pyfaidx import Fasta
import shap.explainers.deep.deep_tf
from deeplift.dinuc_shuffle import dinuc_shuffle
import tensorflow as tf
import os
from utils import one_hot_encode
from importlib import reload
import h5py
import modisco
# sns.set_style('whitegrid')
tf.compat.v1.disable_eager_execution()
reload(shap.explainers.deep.deep_tf)
reload(shap.explainers.deep)
reload(shap.explainers)
reload(shap)
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

genome_file = 'data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa'
overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', index_col=0, nrows=3)
tf_fams = overlap_mat.columns.tolist()


def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
    assert len(list_containing_input_modes_for_an_example) == 1
    onehot_seq = list_containing_input_modes_for_an_example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(50)])

    return [to_return]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
        to_return.append(np.mean(projected_hypothetical_contribs, axis=0))
    return to_return


def prepare_dataset(genome, bed_file, chrom, sample):
    genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    bed = pd.read_csv(filepath_or_buffer=f"data/peaks/{bed_file}", sep="\t", header=None, dtype={0: str})
    bed = bed[bed[0] == str(chrom)]
    seqs = []
    for chrom, start, stop in bed.values:
        midpoint = (start + stop) // 2
        seq = one_hot_encode(genome[chrom][midpoint-125:midpoint+125])
        if seq.shape[0] == 250:
            seqs.append(seq)
    seqs = np.array(seqs)
    return seqs


def modisco_run(tf_name, contribution_scores, hypothetical_scores, one_hots):
    save_file = f"data/modisco/{tf_name}_modisco.hdf5"
    os.system(f'rm -rf {save_file}')

    print('contributions', contribution_scores.shape)
    print('hypothetical contributions', hypothetical_scores.shape)
    print('correct predictions', one_hots.shape)
    # -----------------------Running modisco----------------------------------------------#

    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        # Slight modifications from the default settings
        sliding_window_size=15,
        flank_size=5,
        target_seqlet_fdr=0.15,
        seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=10,
            initial_flank_to_add=2,
            final_flank_to_add=0,
            final_min_cluster_size=30,
            n_cores=5)
    )(
        task_names=['task0'],
        contrib_scores={'task0': contribution_scores},
        hypothetical_contribs={'task0': hypothetical_scores},
        one_hot=one_hots,
        null_per_pos_scores=null_per_pos_scores)

    reload(modisco.util)
    grp = h5py.File(save_file, "w")
    tfmodisco_results.save_hdf5(grp)
    grp.close()
    print(f"Done with {tf_name}")


def get_motifs(bed, genome, chrom, sample):
    tf_name = bed.split('.bed')[0]
    tf_idx = tf_fams.index(tf_name)
    print(f"Processing: {tf_name} class ID: {tf_idx}")

    enc_seqs = prepare_dataset(genome=genome, bed_file=bed, chrom=chrom, sample=sample)
    model = models.load_model(f"saved_models/model_chrom_{chrom}_model.h5")
    preds = model.predict(enc_seqs)
    correct_idx = np.where(preds[:, tf_idx] > 0.5)[0]
    if len(correct_idx) > 10:
        print(f'Total correct preds: {len(correct_idx)}, Total seqs: {enc_seqs.shape[0]}')
        selected_seqs = enc_seqs[correct_idx[:1000], :]
        shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
        shap.explainers.deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers.deep.deep_tf.linearity_1d(0)
        dinuc_shuff_explainer = shap.DeepExplainer(
            (model.input, model.layers[-2].output[:, tf_idx]),
            data=dinuc_shuffle_several_times,
            combine_mult_and_diffref=combine_mult_and_diffref)
        hypothetical_scores = dinuc_shuff_explainer.shap_values(selected_seqs)
        actual_scores = hypothetical_scores * selected_seqs
        print(actual_scores.shape, hypothetical_scores.shape, selected_seqs.shape)
        print(f'Beginning modisco for {bed}-------------------------------------------------\n\n')
        modisco_run(tf_name=tf_name, contribution_scores=actual_scores, hypothetical_scores=hypothetical_scores,
                    one_hots=selected_seqs)


if not os.path.exists("data/modisco"):
    os.mkdir("data/modisco")

num_mods = 0
for bed_file in os.listdir("data/peaks"):
    if not os.path.exists(f"data/modisco/{bed_file.split('.bed')[0]}_modisco.hdf5"):
        get_motifs(bed=bed_file, genome=genome_file, chrom=1, sample=500)
    num_mods += 1
    print(f'Total number of modisco outputs created already: {num_mods}')