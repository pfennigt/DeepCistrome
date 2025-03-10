#! /bin/bash

# creating the required directories
echo "Creating required directories and subdirectories"
mkdir -p data/annotation
mkdir -p data/genome
mkdir -p data/proteome

# Downloading genome
echo "Downloading genome, proteome and Annotation for Arabidopsis thaliana"
wget https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-59/plants/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
wget https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-59/plants/gtf/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.59.gtf.gz
wget https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-59/plants/fasta/arabidopsis_thaliana/pep/Arabidopsis_thaliana.TAIR10.pep.all.fa.gz
mv Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz data/genome
gunzip data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
mv Arabidopsis_thaliana.TAIR10.59.gtf.gz data/annotation
gunzip data/annotation/Arabidopsis_thaliana.TAIR10.59.gtf.gz
mv Arabidopsis_thaliana.TAIR10.pep.all.fa.gz data/genome
gunzip data/proteome/Arabidopsis_thaliana.TAIR10.pep.all.fa.gz

# Downloading RefGen4 Zea mays
echo "Downloading genome of Zea mays. RefGen4"
wget https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-50/plants/fasta/zea_mays/dna/Zea_mays.B73_RefGen_v4.dna.toplevel.fa.gz
mv Zea_mays.B73_RefGen_v4.dna.toplevel.fa.gz data/heat_stress_data_mays
gunzip data/heat_stress_data_mays/Zea_mays.B73_RefGen_v4.dna.toplevel.fa.gz


