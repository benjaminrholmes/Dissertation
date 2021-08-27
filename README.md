## **Using machine learning for thyroid disease prediction and multimorbidity**
___
**Author:** Benjamin R. Holmes

### **Data Cleaning and Preprocessing**
___
Data cleaning and processing was carried out from starting with raw .SAS files 
and performing all data cleaning and pre-processing 
from one python file (**pre-processing.py**)

Python packages used for this step are as follows: 

**pandas** - for data wrangling<br />
**scipy: winsorize** - for flattening out outliers in numerical features <br />
**sklearn: KNNImputer** - for imputing missing values in numerical features <br />
**sklearn: MinMaxScaler** - for normalising numerical and ordered categorical features <br />
**termcolor: colored** - to make console output more readable <br />
___

### **Modelling and Analysis**
___
**Classification task**

Classification modelling was performed using a separate python script for every 
individual dataset/algorithm. Plotting scripts (S*_ROC_CURVES.py) for each dataset were then created that imported 
the relevant python objects from the individual modelling scripts to run them together to generate 
the plots that included multiple algorithms 

Python packages used for this step are as follows: 

**numpy** - for math functions<br />
**pandas** - for data wrangling<br />
**xgboost** - to run gradient boosted tree algorithms <br />
**sklearn:svm** - to run support vector machine algorithms <br />
**sklearn:MLPClassifier** - to run artificial neural network algorithms <br />
**sklearn.metrics: roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, log_loss** - metrics<br />
**sklearn.model_selection: train_test_split, GridSearchCV, StratifiedKFold** - for validation and parameter tuning <br />
**imblearn:** SMOTE - The oversampling technique used <br />
**termcolor: colored** - to make console output more readable <br />
**shap** - For feature importance<br />
**matplotlib** - for plotting<br />

**Regression task**

Regression modelling was performed for all datasets and algorithms from one python script (regression_models.py)
This was possible because there was half the number of separate models to run and it made plotting 
the multi dataset/algorithm figure easier 

Python packages used for this step are as follows: 

**numpy** - for math functions<br />
**pandas** - for data wrangling<br />
**xgboost** - to run gradient boosted tree algorithms <br />
**sklearn:svm** - to run support vector machine algorithms <br />
**sklearn:MLPRegressor** - to run artificial neural network algorithms <br />
**sklearn.metrics: mean_squared_error, r2_score, mean_absolute_error** - metrics <br />
**sklearn.model_selection: train_test_split, GridSearchCV, KFold** - for validation and parameter tuning <br />
**imblearn:** SMOTE - The oversampling technique used <br />
**termcolor: colored** - to make console output more readable <br />
**shap** - For feature importance
**matplotlib** - for plotting
___


### **Genetic Risk Score**
___
Genetic risk score was summed by extracting single nucleotide polymorphisms from genetic data assoaciated with FT4.
Below are the steps in BASH that were carried out for this process. 

The entire process performed using the Whole genome association analysis toolset called PLINK.
https://zzz.bwh.harvard.edu/plink/

**STEP 1**<br />
Create 22 shell scripts using a for loop to make bed files from gene association data in the vcf format
for all 22 chromosomes 

**Example script:** 

for i in {1..22}<br />
do<br />
echo "#!/bin/bash<br />
module load plink<br />
cd example/directory<br />
plink --vcf genefile.chr$i.vcf.gz --make-bed --out example/directory/snp$i">> extract$i.sh<br />
done<br />

This is performed by the **vcftobed_screate.sh** script in the directory 

**STEP 2**<br />
Next is to run those 22 scripts using a for loop

**Example script:** 

for i in {1..22}<br />
do<br />
cd  /example/directory<br />
chmod +x ./extract$i.sh<br />
qsub -b y -V -cwd -l h_vmem=10G -m e ./extract$i.sh<br />
done<br />

This is performed by the **vcftobed_ssub.sh** script in the directory 

**STEP 3**<br />
Next is to create 22 scripts that extracts the SNPs that the user defines by .txt file 
by running another for loop on the bed bim and fam files generated from the previous step

**Example script:** 

for i in {1..22}<br />
do<br />
echo "#!/bin/bash<br />
module load plink<br />
plink --noweb --memory 40000 --bed /example/directory/snp$i.bed --bim /example/directory/snp$i.bim --fam /example/directory/snp$i.fam --extract /example/directory/ft4_snp.txt --make-bed --out /example/directory/ft4_snp$i">> snp_extract$i.sh<br />
done<br />

This is performed by the **snpext_screate.sh** script using the **ft4_snp.txt** for the list of SNPS in the directory 

**STEP 4**<br />
Next is to run those 22 scripts using a for loop

**Example script:** 

for i in {1..22}<br />
do<br />
cd  /example/directory
qsub -b y -V -cwd -l h_vmem=10G -m e ./snp_extract$i.sh<br />
done<br />

This is performed by the **snpext_ssub.sh** script in the directory 

**STEP 5**<br />
Next is to merge all the extracted bed files using a list of those files in a .txt 

**Example script:** 

module load plink<br />

plink --merge-list ./allfiles.txt --make-bed --out ./merged_snp<br />

This is performed by the **snpmerge.sh** script and the list of bed files **allfiles.txt** in the directory 

**STEP 6**<br />
Next is to sum the merged files and generate a genetic risk score that corresponds with a participant ID

**Example script:** 

module load plink<br />

plink --bfile ./merged_snp --score ./scores.txt sum --score-no-mean-imputation --out ./ft4_score<br />

This is performed by the **snpscore.sh** and the **scores.txt** in the directory 

**STEP 7**<br />
The last step is to merge the output file which is in the .profile file format with the existing participant 
data by participant ID.

**This was done using python code that can be found in pre_processing.py in the function called read_SNP and merge_SNP** 


___