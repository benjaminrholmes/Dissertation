#!/bin/bash

for i in {1..22}
do
cd  /data/WHRI-Marouli/bt20732/SNP/TREND/submission_scripts
chmod +x ./extract$i.sh
qsub -b y -V -cwd -l h_vmem=10G -m e ./extract$i.sh
done
