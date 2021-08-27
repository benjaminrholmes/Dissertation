#!/bin/bash
#$ -cwd           # Set the working directory for the job to the current directory
#$ -pe smp 1      # Request 1 core
#$ -l h_rt=1:0:0  # Request 1 hour runtime
#$ -l h_vmem=1G   # Request 1GB RAM
#$ -m e

for i in {1..22}
do
echo "#!/bin/bash
module load plink
plink --noweb --memory 40000 --bed /data/WHRI-Marouli/bt20732/SNP/TREND/data/snp$i.bed --bim /data/WHRI-Marouli/bt20732/SNP/TREND/data/snp$i.bim --fam /data/WHRI-Marouli/bt20732/SNP/TREND/data/snp$i.fam --extract /data/WHRI-Marouli/bt20732/SNP/TREND/data/ft4_snp.txt --make-bed --out /data/WHRI-Marouli/bt20732/SNP/TREND/data/ft4_snp$i">> snp_extract$i.sh
done
