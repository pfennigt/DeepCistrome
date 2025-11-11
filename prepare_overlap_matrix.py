from utils import peak_overlap_genome_bins
import pandas as pd
pd.options.display.width = 0

genome_path = 'data/genome/Zea_mays.B73_RefGen_v4.dna.toplevel.fa'
peak_overlap_genome_bins(genome=genome_path, bin_size=250, frac=0.7)

overlap_matrix = pd.read_csv(filepath_or_buffer='data/overlap.bed', sep='\t', header=None,
                             dtype={0: str, 1: int, 2: int, 3: str, 4: int})
overlap_matrix[3] = overlap_matrix[3].str.replace('data/peaks/', '').str.replace('.bed', '')
overlap_matrix[4] = [1 if v > 0 else v for v in overlap_matrix[4]]
overlap_matrix[5] = [f"{v[0]}:{v[1]}:{v[2]}" for v in overlap_matrix.values]
overlap_matrix = overlap_matrix.pivot(columns=3, index=5, values=4)
overlap_matrix = overlap_matrix[overlap_matrix.sum(axis=1) > 0]
overlap_matrix.to_csv(path_or_buf='data/overlap_matrix.bed', sep='\t')
