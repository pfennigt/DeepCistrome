#!/usr/bin/env python3

import argparse
import os
import sys
from pyfaidx import Fasta
import gffutils
import textwrap

def reverse_complement(seq):
    """Computes the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
                  'N': 'N', 'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
                  'n': 'n'}
    return "".join(complement.get(base, base) for base in reversed(seq))

def extract_upstream_regions(fasta_file, gff_file, output_file,
                             upstream_dist=1000, downstream_dist=500,
                             feature_type='gene'):
    """
    Extracts upstream regions of specified features (default: genes)
    from a genome FASTA file based on a GFF annotation file.

    The extracted region is defined relative to the feature's start coordinate
    and maintains the feature's orientation.

    Args:
        fasta_file (str): Path to the genome FASTA file.
        gff_file (str): Path to the GFF3 annotation file.
        output_file (str): Path to the output FASTA file for extracted regions.
        upstream_dist (int): Distance upstream from the feature start to include.
        downstream_dist (int): Distance downstream from the feature start to include.
        feature_type (str): The GFF feature type to process (e.g., 'gene').
    """
    print(f"Loading genome from: {fasta_file}")
    try:
        genome = Fasta(fasta_file)
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta_file}", file=sys.stderr)
        sys.exit(1)

    gff_db_file = gff_file + '.db'
    if os.path.exists(gff_db_file):
        print(f"Using existing GFF database: {gff_db_file}")
        try:
            db = gffutils.FeatureDB(gff_db_file)
        except Exception as e:
             print(f"Error opening GFF database {gff_db_file}: {e}", file=sys.stderr)
             print("Attempting to rebuild database...")
             try:
                 os.remove(gff_db_file) # Remove potentially corrupt db
                 db = gffutils.create_db(gff_file, dbfn=gff_db_file, force=True,
                                         keep_order=True, merge_strategy='merge',
                                         sort_attribute_values=True)
             except Exception as e_create:
                 print(f"Error creating GFF database for {gff_file}: {e_create}", file=sys.stderr)
                 sys.exit(1)

    else:
        print(f"Creating GFF database for: {gff_file} (this may take a while)...")
        try:
            db = gffutils.create_db(gff_file, dbfn=gff_db_file, force=True,
                                    keep_order=True, merge_strategy='merge',
                                    sort_attribute_values=True)
            print(f"Database created: {gff_db_file}")
        except Exception as e:
            print(f"Error creating GFF database for {gff_file}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Extracting {upstream_dist}bp upstream and {downstream_dist}bp downstream "
          f"regions for features of type '{feature_type}'...")

    count = 0
    skipped_coords = 0
    skipped_chrom = 0
    try:
        with open(output_file, 'w') as outfile:
            for feature in db.features_of_type(feature_type, order_by=('seqid', 'start')):
                chrom = feature.chrom
                strand = feature.strand
                gene_id = feature.attributes.get('ID', [f'{feature_type}_{feature.start}'])[0] # Get ID or generate one

                if chrom not in genome:
                    print(f"Warning: Chromosome '{chrom}' from GFF not found in FASTA file. Skipping feature {gene_id}.", file=sys.stderr)
                    skipped_chrom += 1
                    continue

                chrom_len = len(genome[chrom])

                # Calculate coordinates based on strand
                if strand == '+':
                    # Gene start is feature.start
                    start_1based = feature.start - upstream_dist
                    end_1based = feature.start + downstream_dist -1 # Adjust as window includes start
                elif strand == '-':
                    # Gene start is feature.end (biologically)
                    # Upstream region is towards higher coordinates
                    start_1based = feature.end - downstream_dist + 1 # Adjust as window includes end
                    end_1based = feature.end + upstream_dist
                else:
                     print(f"Warning: Feature {gene_id} on chromosome {chrom} has unspecified strand ('{strand}'). Skipping.", file=sys.stderr)
                     continue


                # --- Boundary Checks ---
                original_start = start_1based
                original_end = end_1based

                # Adjust start if it goes below 1
                if start_1based < 1:
                    start_1based = 1

                # Adjust end if it exceeds chromosome length
                if end_1based > chrom_len:
                    end_1based = chrom_len

                # Check if coordinates are still valid after adjustments
                if start_1based >= end_1based:
                    print(f"Warning: Invalid coordinates after boundary check for {gene_id} "
                          f"(Chr: {chrom}, Strand: {strand}, "
                          f"Original Calc: {original_start}-{original_end}, "
                          f"Adjusted: {start_1based}-{end_1based}). Skipping.", file=sys.stderr)
                    skipped_coords += 1
                    continue

                # --- Extract Sequence (using 0-based indexing for pyfaidx) ---
                # pyfaidx slices are [start, end), so end coordinate is exclusive
                seq_0based_start = start_1based - 1
                seq_0based_end = end_1based

                try:
                    sequence = genome[chrom][seq_0based_start:seq_0based_end].seq
                except (IndexError, ValueError) as e:
                     print(f"Warning: Error extracting sequence for {gene_id} "
                           f"(Chr: {chrom}, Coords: {start_1based}-{end_1based}, Strand: {strand}). "
                           f"Error: {e}. Skipping.", file=sys.stderr)
                     skipped_coords += 1
                     continue


                # --- Orient Sequence ---
                if strand == '-':
                    sequence = reverse_complement(sequence)

                # --- Write Output ---
                header = f">{gene_id}_upstream region={chrom}:{start_1based}-{end_1based} strand={strand} " \
                         f"length={len(sequence)}"
                outfile.write(header + '\n')
                # Wrap sequence lines
                for i in range(0, len(sequence), 60):
                     outfile.write(sequence[i:i+60] + '\n')
                count += 1

                if count % 1000 == 0:
                    print(f"Processed {count} features...")

    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred: {e}", file=sys.stderr)
         sys.exit(1)


    print(f"\nExtraction complete.")
    print(f"Successfully extracted sequences for {count} features.")
    if skipped_chrom > 0:
         print(f"Skipped {skipped_chrom} features due to missing chromosomes in FASTA.")
    if skipped_coords > 0:
        print(f"Skipped {skipped_coords} features due to invalid coordinates after boundary checks or extraction errors.")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract gene upstream regions (+/- strand aware) from a genome FASTA file using GFF3 annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fasta_file", help="Path to the genome FASTA file (.fa, .fasta)")
    parser.add_argument("gff_file", help="Path to the GFF3 annotation file (.gff, .gff3)")
    parser.add_argument("output_file", help="Path for the output FASTA file")
    parser.add_argument("-u", "--upstream", type=int, default=1000,
                        help="Distance upstream of the gene start coordinate to extract.")
    parser.add_argument("-d", "--downstream", type=int, default=500,
                        help="Distance downstream of the gene start coordinate to extract.")
    parser.add_argument("-t", "--type", type=str, default="gene",
                        help="GFF feature type to extract upstream regions for (e.g., 'gene', 'mRNA').")

    args = parser.parse_args()

    # Basic input validation
    if not os.path.exists(args.fasta_file):
         print(f"Error: Input FASTA file not found: {args.fasta_file}", file=sys.stderr)
         sys.exit(1)
    if not os.path.exists(args.gff_file):
         print(f"Error: Input GFF file not found: {args.gff_file}", file=sys.stderr)
         sys.exit(1)
    if args.upstream < 0 or args.downstream < 0:
         print(f"Error: Upstream and downstream distances cannot be negative.", file=sys.stderr)
         sys.exit(1)


    extract_upstream_regions(args.fasta_file, args.gff_file, args.output_file,
                             args.upstream, args.downstream, args.type)
