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
cd /data/WHRI-Marouli/DATA/SHIP/
plink --vcf SHIP_2020_41_D-SHIP-0_R4a.chr$i.dose.vcf.gz --make-bed --out /data/WHRI-Marouli/bt20732/SNP/SHIP/data/snp$i">> extract$i.sh
done
