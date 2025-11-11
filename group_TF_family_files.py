#!/usr/bin/env python3
"""
Organize .narrowPeak files into family folders based on a TSV mapping.

TSV format (tab-separated):
TF    v5_geneID    v3_geneID    family    sub-family

For each TF, this script looks for files named "<TF>*.narrowPeak"
in the INPUT_DIR and moves them into a folder named after the 'family' column
inside OUTPUT_PARENT_DIR (creating it if necessary).
"""

import os
import shutil
import pandas as pd
from glob import glob

# === CONFIGURATION (adjust these paths) ===
TSV_FILE = "data/zea_mays/Galli2025/GEM3_metadata.csv"          # Path to your TSV file
INPUT_DIR = "data/zea_mays/Galli2025/GEM3_narrowPeak_unified/Mo17_GEM3_narrowPeak_Unified"         # Directory containing the .narrowPeak files
OUTPUT_PARENT_DIR = "data/zea_mays/Galli2025/peaks"   # Parent folder to hold family subfolders
# ==========================================


def main():
    # Load TSV file
    df = pd.read_csv(TSV_FILE, sep=",")

    # Ensure output directory exists
    os.makedirs(OUTPUT_PARENT_DIR, exist_ok=True)

    moved_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        tf_name = str(row["TF"]).strip()
        family = str(row["family"]).strip()

        # Create family folder if it doesn't exist
        family_dir = os.path.join(OUTPUT_PARENT_DIR, family)
        os.makedirs(family_dir, exist_ok=True)

        # Find matching .narrowPeak files
        pattern = os.path.join(INPUT_DIR, f"{tf_name}*.narrowPeak")
        matches = glob(pattern)

        if not matches:
            print(f"[WARNING] No file found for TF '{tf_name}'")
            missing_count += 1
            continue

        for match in matches:
            filename = os.path.basename(match)
            destination = os.path.join(family_dir, filename)
            shutil.move(match, destination)
            print(f"[MOVED] {filename} → {family}/")
            moved_count += 1

    print("\n=== SUMMARY ===")
    print(f"Files moved: {moved_count}")
    print(f"TFs with missing files: {missing_count}")


if __name__ == "__main__":
    main()