#!/bin/bash
#$ -cwd           # Set the working directory for the job to the current directory
#$ -pe smp 1      # Request 1 core
#$ -l h_rt=1:0:0  # Request 1 hour runtime
#$ -l h_vmem=1G   # Request 1GB RAM
#$ -m e

module load plink

plink --bfile ./merged_snp --score ./scores.txt sum --score-no-mean-imputation --out ./ft4_score