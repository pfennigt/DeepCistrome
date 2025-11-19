#!/usr/bin/env python3

import argparse
import sys
from pyfaidx import Fasta

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extracts genomic regions from a FASTA file based on features in a GFF3 file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-g", "--gff",
        metavar="<file>",
        type=str,
        required=True,
        help="Path to the input GFF3 file."
    )
    
    parser.add_argument(
        "-f", "--fasta",
        metavar="<file>",
        type=str,
        required=True,
        help="Path to the input FASTA file (must be indexed with `samtools faidx`)."
    )
    
    parser.add_argument(
        "-o", "--output",
        metavar="<file>",
        type=str,
        required=True,
        help="Path for the output FASTA file."
    )
    
    parser.add_argument(
        "--feature",
        metavar="<type>",
        type=str,
        default="gene",
        help="The feature type to extract from the GFF's 3rd column (default: 'gene')."
    )
    
    parser.add_argument(
        "--flank",
        metavar="<int>",
        type=int,
        default=0,
        help="Integer value to add/subtract from feature start/end (default: 0)."
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()

    try:
        # Load the FASTA file using pyfaidx. It's very fast.
        genome = Fasta(args.fasta)
        
    except FileNotFoundError:
        print(f"Error: FASTA file not found at '{args.fasta}'", file=sys.stderr)
        sys.exit(1)

    print(f"📖 Parsing features of type '{args.feature}' from '{args.gff}'...")
    
    # Open GFF and output FASTA file
    try:
        with open(args.gff, 'r') as gff_file, open(args.output, 'w') as out_fasta:
            count = 0
            for line in gff_file:
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                
                # Ensure the line has enough columns
                if len(parts) < 8:
                    continue
                
                # Check if the feature type matches
                if parts[2] == args.feature:
                    try:
                        chrom = parts[0]
                        start = int(parts[3])
                        end = int(parts[4])
                        
                        # Apply flanking regions
                        new_start = max(1, start - args.flank)
                        new_end = end + args.flank
                        
                        # Get sequence using pyfaidx.
                        # pyfaidx uses 0-based indexing, GFF is 1-based.
                        # seq = genome[chrom][start-1:end]
                        sequence = genome[chrom][new_start - 1:new_end]
                        
                        # Write the new FASTA entry
                        header = f">{sequence.long_name}"
                        out_fasta.write(f"{header}\n{sequence.seq}\n")
                        count += 1
                        
                    except (ValueError, KeyError) as e:
                        # Handle errors like non-integer coordinates or missing chromosomes
                        print(f"Warning: Skipping invalid line or chromosome: {line.strip()} | Error: {e}", file=sys.stderr)
                        continue
                        
            print(f"✅ Successfully extracted {count} sequences to '{args.output}'.")

    except FileNotFoundError:
        print(f"Error: GFF file not found at '{args.gff}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
