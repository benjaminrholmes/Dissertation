import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from termcolor import colored

pd.set_option('mode.chained_assignment', None)


print(colored("\n...........DATA CLEANING & PRE-PROCESSING STARTED............", "green"))


def read_raw_sas(dir):
    raw = pd.read_sas(dir)
    return raw


SHIP0 = read_raw_sas(
    "D:\\Files\\Work\\Pycharm Projects\\Dissertation\\raw_data\\SHIP-0\\ship_2020_41_d_s0_20200904.sas7bdat")
SHIP1 = read_raw_sas(
    "D:\\Files\\Work\\Pycharm Projects\\Dissertation\\raw_data\\SHIP-1\\ship_2020_41_d_s1_20200904.sas7bdat")
SHIP2 = read_raw_sas(
    "D:\\Files\\Work\\Pycharm Projects\\Dissertation\\raw_data\\SHIP-2\\ship_2020_41_d_s2_20200904.sas7bdat")
SHIP3 = read_raw_sas(
    "D:\\Files\\Work\\Pycharm Projects\\Dissertation\\raw_data\\SHIP-3\\ship_2020_41_d_s3_20200904.sas7bdat")
TREND0 = read_raw_sas(
    "D:\\Files\\Work\\Pycharm Projects\\Dissertation\\raw_data\\TREND-0\\ship_2020_41_d_t0_20200904.sas7bdat")


def read_labels(dir):
    labels = pd.read_csv(dir)
    return labels


SHIP0_labels = read_labels("D:\\Files\\Work\\Pycharm Projects\Dissertation\\raw_data\\SHIP-0\\S0_labels.csv")
SHIP1_labels = read_labels("D:\\Files\\Work\\Pycharm Projects\Dissertation\\raw_data\\SHIP-1\\S1_labels.csv")
SHIP2_labels = read_labels("D:\\Files\\Work\\Pycharm Projects\Dissertation\\raw_data\\SHIP-2\\S2_labels.csv")
SHIP3_labels = read_labels("D:\\Files\\Work\\Pycharm Projects\Dissertation\\raw_data\\SHIP-3\\S3_labels.csv")
TREND0_labels = read_labels("D:\\Files\\Work\\Pycharm Projects\Dissertation\\raw_data\\TREND-0\\T0_labels.csv")


def rename_features(data, labels):
    rename_dict = dict(zip(labels.original, labels.translated))
    new_data = data.rename(rename_dict, axis=1)
    return new_data


SHIP0 = rename_features(SHIP0, SHIP0_labels)
SHIP1 = rename_features(SHIP1, SHIP1_labels)
SHIP2 = rename_features(SHIP2, SHIP2_labels)
SHIP3 = rename_features(SHIP3, SHIP3_labels)
TREND0 = rename_features(TREND0, TREND0_labels)

# Function to read in the SNP data
def read_SNP(dir):
    SNP = pd.read_csv(dir)
    SNP = SNP.drop(["IID", "PHENO", "CNT", "CNT2"], axis=1)
    SNP = SNP.rename({"FID": "PID"}, axis=1)
    return SNP


SHIP_SNP = read_SNP("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\snp_score\\SHIP\\results\\SHIP_ft4_snp_scores.csv")
TREND_SNP = read_SNP("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\snp_score\\TREND\\results\\TREND_ft4_snp_scores.csv")

# Function to merge the SNP data with participant data.
def merge_SNP(data, SNP):
    print("Before SNP merge: ", data.shape)
    SNP_df = pd.merge(data, SNP, on="PID")
    print("After SNP merge: ", SNP_df.shape)
    return SNP_df

print(colored("\nSNP MERGING............", "blue"))
print(colored("\nSHIP0", "yellow"))
SHIP0 = merge_SNP(SHIP0, SHIP_SNP)
print(colored("\nSHIP1", "yellow"))
SHIP1 = merge_SNP(SHIP1, SHIP_SNP)
print(colored("\nSHIP2", "yellow"))
SHIP2 = merge_SNP(SHIP2, SHIP_SNP)
print(colored("\nSHIP3", "yellow"))
SHIP3 = merge_SNP(SHIP3, SHIP_SNP)
print(colored("\nTREND0", "yellow"))
TREND0 = merge_SNP(TREND0, TREND_SNP)

# function to create list of numerical features
def create_num_names(labels):
    num_col_names = labels.num_feat.tolist()
    num_col_list = [num_col_names for num_col_names in num_col_names if
                    str(num_col_names) != 'nan']
    return num_col_list


SHIP0_num = create_num_names(SHIP0_labels)
SHIP1_num = create_num_names(SHIP1_labels)
SHIP2_num = create_num_names(SHIP2_labels)
SHIP3_num = create_num_names(SHIP3_labels)
TREND0_num = create_num_names(TREND0_labels)


# function that removes unwanted features
def data_cleaner(data, labels):
    excluded_col_names = labels.excl_feat.tolist()
    excl_list = [excluded_col_names for excluded_col_names in excluded_col_names if
                         str(excluded_col_names) != 'nan']
    char_col_names = data.select_dtypes(include=["object", "datetime"]).columns.tolist()
    print("df original: ", data.shape)
    df = data.drop(excl_list, axis=1)
    print("df after features removed from excluded list: ", df.shape)
    df2 = df.drop(char_col_names, axis=1)
    print("df after character and date features removed", df2.shape)
    return df2

# function that imputes features and removes features that are above a missingness threshold
def data_imputer(data, numf_list, pm, nn):
    imputer = KNNImputer(n_neighbors=nn)
    participants = data.shape[0]
    na_sum = data[numf_list].isna().sum().to_frame()
    na_sum.loc[(na_sum.iloc[:, 0] / participants) > pm, 'Impute'] = 0
    na_sum.loc[(na_sum.iloc[:, 0] / participants) <= pm, 'Impute'] = 1
    imp_list = na_sum.index[na_sum["Impute"] == 1].tolist()
    print("Number of features imputed: ", len(imp_list))
    drop_list = na_sum.index[na_sum["Impute"] == 0].tolist()
    print("Number of features removed due to exceeding missingness threshold: ", len(drop_list))
    data[imp_list] = pd.DataFrame(imputer.fit_transform(data[imp_list]), columns=data[imp_list].columns)
    df = data.drop(drop_list, axis=1)
    return df

# function to winsorise numerical features
def data_winsorize(data, labels, ll, ul):
    win_col_names = labels.win.tolist()
    win_col_list = [win_col_names for win_col_names in win_col_names if
                    str(win_col_names) != 'nan']
    for feature in win_col_list:
        data[feature] = winsorize(data[feature], limits=[ll, ul])
    return data

# Create BMI feature
SHIP1["BMI"] = SHIP1["SOMA_BW"] / ((SHIP1["SOMA_HT"]/100) ** 2)
SHIP2["BMI"] = SHIP2["SOMA_BW"] / ((SHIP2["SOMA_HT"]/100) ** 2)
SHIP3["BMI"] = SHIP3["SOMA_BW"] / ((SHIP3["SOMA_HT"]/100) ** 2)

print(colored("\nDESCRIPTIVE STATISTICS............", "blue"))

print(colored("\nAGE", "yellow"))
print("SHIP-0 age: ", SHIP0["AGE"].describe())
print("SHIP-1 age: ", SHIP1["AGE"].describe())
print("SHIP-2 age: ", SHIP2["AGE"].describe())
print("SHIP-3 age: ", SHIP3["AGE"].describe())
print("TREND-0 age: ", TREND0["AGE"].describe())

print(colored("\nSEX", "yellow"))
print("SHIP-0 sex: ")
print(SHIP0["SEX"].value_counts(normalize=True))
print("SHIP-1 sex: ")
print(SHIP1["SEX"].value_counts(normalize=True))
print("SHIP-2 sex: ")
print(SHIP2["SEX"].value_counts(normalize=True))
print("SHIP-3 sex: ")
print(SHIP3["SEX"].value_counts(normalize=True))
print("TREND-0 sex: ")
print(TREND0["SEX"].value_counts(normalize=True))

print(colored("\nTSH", "yellow"))
print("SHIP-0 TSH: ", SHIP0["TSH"].describe())
print("SHIP-1 TSH: ", SHIP1["TSH"].describe())
print("SHIP-2 TSH: ", SHIP2["TSH"].describe())
print("SHIP-3 TSH: ", SHIP3["TSH"].describe())
print("TREND-0 TSH: ", TREND0["TSH"].describe())

print(colored("\nFT4", "yellow"))
print("SHIP-0 FT4: ", SHIP0["FT4"].describe())
print("SHIP-1 FT4: ", SHIP1["FT4"].describe())
print("SHIP-2 FT4: ", SHIP2["FT4"].describe())
print("SHIP-3 FT4: ", SHIP3["FT4"].describe())
print("TREND-0 FT4: ", TREND0["FT4"].describe())

print(colored("\nFT3", "yellow"))
print("SHIP-0 FT3: ", SHIP0["FT3"].describe())
print("SHIP-1 FT3: ", SHIP1["FT3_RE"].describe())
print("SHIP-2 FT3: ", SHIP2["FT3_RE"].describe())
print("SHIP-3 FT3: ", SHIP3["FT3"].describe())
print("TREND-0 FT3: ", TREND0["FT3_RE"].describe())

print(colored("\nBMI", "yellow"))
print("SHIP-0 BMI: ", SHIP0["BMI"].describe())
print("SHIP-1 BMI: ", SHIP1["BMI"].describe())
print("SHIP-2 BMI: ", SHIP2["BMI"].describe())
print("SHIP-3 BMI: ", SHIP3["BMI"].describe())
print("TREND-0 BMI: ", TREND0["BMI"].describe())

print(colored("\nWaist circumference", "yellow"))
print("SHIP-0 WC: ", SHIP0["SOMA_WC"].describe())
print("SHIP-1 WC: ", SHIP1["SOMA_WC"].describe())
print("SHIP-2 WC: ", SHIP2["SOMA_WC"].describe())
print("SHIP-3 WC: ", SHIP3["SOMA_WC"].describe())
print("TREND-0 WC: ", TREND0["SOMA_WC"].describe())


print(colored("\nCLEANING AND PRE-PROCESSING THE DATA............", "blue"))
print(colored("\nSHIP0", "yellow"))

SHIP0 = data_imputer(SHIP0, SHIP0_num, 0.15, 10)
SHIP0 = data_cleaner(SHIP0, SHIP0_labels)
SHIP0 = data_winsorize(SHIP0, SHIP0_labels, 0.001, 0.001)

print(colored("\nSHIP1", "yellow"))
SHIP1 = data_imputer(SHIP1, SHIP1_num, 0.15, 10)
SHIP1 = data_cleaner(SHIP1, SHIP1_labels)
SHIP1 = data_winsorize(SHIP1, SHIP1_labels, 0.001, 0.001)

print(colored("\nSHIP2", "yellow"))
SHIP2 = data_imputer(SHIP2, SHIP2_num, 0.15, 10)
SHIP2 = data_cleaner(SHIP2, SHIP2_labels)
SHIP2 = data_winsorize(SHIP2, SHIP2_labels, 0.001, 0.001)

print(colored("\nSHIP3", "yellow"))
SHIP3 = data_imputer(SHIP3, SHIP3_num, 0.15, 10)
SHIP3 = data_cleaner(SHIP3, SHIP3_labels)
SHIP3 = data_winsorize(SHIP3, SHIP3_labels, 0.001, 0.001)

print(colored("\nTREND0", "yellow"))
TREND0 = data_imputer(TREND0, TREND0_num, 0.15, 10)
TREND0 = data_cleaner(TREND0, TREND0_labels)
TREND0 = data_winsorize(TREND0, TREND0_labels, 0.001, 0.001)

# SHIP0 preprocessing
print(colored("\nREMOVING PREGNANT PARTICIPANTS............", "blue"))
print(colored("\nSHIP0", "yellow"))
# add integer coding for interview question NaNs
SHIP0.loc[(SHIP0.INT_THY1 == 2) & (SHIP0.INT_THY2.isnull()), "INT_THY2"] = 0
SHIP0.loc[(SHIP0.INT_THY1 == 2) & (SHIP0.INT_THY3.isnull()), "INT_THY3"] = 0

print("before removing preg participants", SHIP0.shape)
#
# removing currently pregnant participants
SHIP0 = SHIP0.drop(SHIP0[SHIP0.BLT_PREG == 1].index)

SHIP0 = SHIP0.drop(SHIP0[SHIP0.BLT_PREG == 2].index)

print("after removing preg participants", SHIP0.shape)

# removing currently_preg and radio_therapy_year features from dataset
SHIP0 = SHIP0.drop(["BLT_PREG"], axis=1)
print("removing preg feature", SHIP0.shape)

# standardise binary coding variables
SHIP0['SEX'] = SHIP0['SEX'].replace(1, 0).replace(2, 1)
SHIP0['INT_THY1'] = SHIP0['INT_THY1'].replace(2, 0).replace(8, 0).replace(9, 0)
SHIP0['INT_THY2'] = SHIP0['INT_THY2'].replace(2, 0).replace(8, 0)
SHIP0['INT_THY3'] = SHIP0['INT_THY3'].replace(2, 0).replace(8, 0)
SHIP0['MED_7D'] = SHIP0['MED_7D'].replace(2, 0).replace(8, 0).replace(9, 0)

# Convert not assessed and unlevied to 0
SHIP0['THY_ER'] = SHIP0['THY_ER'].replace(8, 0).replace(9, 0)
SHIP0['THY_EL'] = SHIP0['THY_EL'].replace(8, 0).replace(9, 0)
SHIP0['THY_FIN'] = SHIP0['THY_FIN'].replace(8, 0).replace(9, 0)

# fill in NA values with 0
SHIP0 = SHIP0.fillna(0)

# SHIP1 preprocessing

print(colored("\nSHIP1", "yellow"))

# add integer coding for interview question NaNs
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_HYPER.isnull()), "INT_HYPER"] = 0
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_HYPO.isnull()), "INT_HYPO"] = 0
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_STRUMA.isnull()), "INT_STRUMA"] = 0
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_LUMP.isnull()), "INT_LUMP"] = 0
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_GOIT.isnull()), "INT_GOIT"] = 0
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_CARC.isnull()), "INT_CARC"] = 0
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_OTH.isnull()), "INT_OTH"] = 0
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_DNK.isnull()), "INT_DNK"] = 0
SHIP1.loc[(SHIP1.INT_THY == 2) & (SHIP1.INT_RTR.isnull()), "INT_RTR"] = 0

print("before removing preg participants", SHIP1.shape)
# removing currently pregnant participants
SHIP1 = SHIP1.drop(SHIP1[SHIP1.BLT_PREG == 1].index)
SHIP1 = SHIP1.drop(SHIP1[SHIP1.BLT_PREG == 8].index)
print("after removing preg participants", SHIP1.shape)

# removing currently_preg and radio_therapy_year features from dataset
SHIP1 = SHIP1.drop(["BLT_PREG"], axis=1)
print("removing preg feature", SHIP1.shape)
# standardise binary coding variables
SHIP1['SEX'] = SHIP1['SEX'].replace(1, 0).replace(2, 1)
SHIP1['INT_THY'] = SHIP1['INT_THY'].replace(2, 0)
SHIP1['INT_THY'] = SHIP1['INT_THY'].replace(998, 0)

# Convert not assessed and unlevied to 0
SHIP1['THR_NNR'] = SHIP1['THR_NNR'].replace(8, 0).replace(9, 0)
SHIP1['THR_NNL'] = SHIP1['THR_NNL'].replace(8, 0).replace(9, 0)
SHIP1['THY_ER'] = SHIP1['THY_ER'].replace(8, 0).replace(9, 0)
SHIP1['THY_EL'] = SHIP1['THY_EL'].replace(8, 0).replace(9, 0)
SHIP1['THY_FIN'] = SHIP1['THY_FIN'].replace(8, 0).replace(9, 0)

SHIP1['INT_SMOK'] = SHIP1['INT_SMOK'].replace(999, 0).replace(998, 0).replace(3, 1)

# fill in NA values with 0
SHIP1 = SHIP1.fillna(0)

#### SHIP2 ####
# add integer coding for interview question NaNs
SHIP2.loc[(SHIP2.INT_THY == 2) & (SHIP2.INT_HYPER.isnull()), "INT_HYPER"] = 0
SHIP2.loc[(SHIP2.INT_THY == 2) & (SHIP2.INT_HYPO.isnull()), "INT_HYPO"] = 0
SHIP2.loc[(SHIP2.INT_THY == 2) & (SHIP2.INT_RIT.isnull()), "INT_RIT"] = 0
SHIP2.loc[(SHIP2.INT_THY == 2) & (SHIP2.INT_GOIT.isnull()), "INT_GOIT"] = 0
SHIP2.loc[(SHIP2.INT_THY == 2) & (SHIP2.INT_NODE.isnull()), "INT_NODE"] = 0
SHIP2.loc[(SHIP2.INT_THY == 2) & (SHIP2.INT_OTH.isnull()), "INT_OTH"] = 0
SHIP2.loc[(SHIP2.INT_THY == 2) & (SHIP2.INT_DNK.isnull()), "INT_DNK"] = 0
SHIP2.loc[(SHIP2.INT_THY == 2) & (SHIP2.INT_RTR.isnull()), "INT_RTR"] = 0


# standardise binary coding variables
SHIP2['SEX'] = SHIP2['SEX'].replace(1, 0).replace(2, 1)
SHIP2['INT_THY'] = SHIP2['INT_THY'].replace(2, 0)
SHIP2['INT_THY'] = SHIP2['INT_THY'].replace(998, 0)
SHIP2['INT_RIT'] = SHIP2['INT_RIT'].replace(2, 1)
SHIP2['INT_HYPER'] = SHIP2['INT_HYPER'].replace(-1, 1)
SHIP2['INT_HYPO'] = SHIP2['INT_HYPO'].replace(-1, 1)
SHIP2['INT_GOIT'] = SHIP2['INT_GOIT'].replace(-1, 1)
SHIP2['INT_NODE'] = SHIP2['INT_NODE'].replace(-1, 1)
SHIP2['INT_OTH'] = SHIP2['INT_OTH'].replace(-1, 1)
SHIP2['INT_DNK'] = SHIP2['INT_DNK'].replace(-1, 1)
SHIP2['INT_RTR'] = SHIP2['INT_RTR'].replace(-1, 1)
SHIP2['INT_SMOK'] = SHIP2['INT_SMOK'].replace(2, 0).fillna(0)
#
# Convert not assessed and unlevied to 0
SHIP2['THR_NNR'] = SHIP2['THR_NNR'].replace(8, 0).replace(9, 0)
SHIP2['THR_NNL'] = SHIP2['THR_NNL'].replace(8, 0).replace(9, 0)
SHIP2['THY_ER'] = SHIP2['THY_ER'].replace(8, 0).replace(9, 0)
SHIP2['THY_EL'] = SHIP2['THY_EL'].replace(8, 0).replace(9, 0)
SHIP2['THY_FIN'] = SHIP2['THY_FIN'].replace(8, 0).replace(9, 0)
SHIP2['THY_HR'] = SHIP2['THY_HR'].replace(8, 0).replace(9, 0)
SHIP2['THY_HL'] = SHIP2['THY_HL'].replace(8, 0).replace(9, 0)



# fill in NA values with 0
SHIP2 = SHIP2.fillna(0)

#### SHIP3 ####

SHIP3['INT_THY'] = SHIP3['INT_THY'].replace(998, 0)

SHIP3.loc[(SHIP3.INT_THY == 0) & (SHIP3.INT_HYPER.isnull()), "INT_HYPER"] = 0
SHIP3.loc[(SHIP3.INT_THY == 0) & (SHIP3.INT_HYPO.isnull()), "INT_HYPO"] = 0
SHIP3.loc[(SHIP3.INT_THY == 0) & (SHIP3.INT_GOIT.isnull()), "INT_GOIT"] = 0
SHIP3.loc[(SHIP3.INT_THY == 0) & (SHIP3.INT_NODE.isnull()), "INT_NODE"] = 0
SHIP3.loc[(SHIP3.INT_THY == 0) & (SHIP3.INT_DNK.isnull()), "INT_DNK"] = 0
SHIP3.loc[(SHIP3.INT_THY == 0) & (SHIP3.INT_RTR.isnull()), "INT_RTR"] = 0
#
#
# # standardise binary coding variables
SHIP3['SEX'] = SHIP3['SEX'].replace(1, 0).replace(2, 1)

#
# Convert not assessed and unlevied to 0
SHIP3['THR_NNR'] = SHIP3['THR_NNR'].replace(8, 0).replace(9, 0)
SHIP3['THR_NNL'] = SHIP3['THR_NNL'].replace(8, 0).replace(9, 0)
SHIP3['THR_NNI'] = SHIP3['THR_NNI'].replace(8, 0).replace(9, 0)
SHIP3['THY_ER'] = SHIP3['THY_ER'].replace(8, 0).replace(9, 0)
SHIP3['THY_EL'] = SHIP3['THY_EL'].replace(8, 0).replace(9, 0)
SHIP3['THY_FIN'] = SHIP3['THY_FIN'].replace(8, 0).replace(9, 0)
SHIP3['THY_HR'] = SHIP3['THY_HR'].replace(8, 0).replace(9, 0)
SHIP3['THY_HL'] = SHIP3['THY_HL'].replace(8, 0).replace(9, 0)


# fill in NA values with 0
SHIP3 = SHIP3.fillna(0)

print(colored("\nTREND0", "yellow"))

# add integer coding for interview question NaNs
TREND0.loc[(TREND0.INT_THY == 2) & (TREND0.INT_HYPER.isnull()), "INT_HYPER"] = 0
TREND0.loc[(TREND0.INT_THY == 2) & (TREND0.INT_HYPO.isnull()), "INT_HYPO"] = 0
TREND0.loc[(TREND0.INT_THY == 2) & (TREND0.INT_GOIT.isnull()), "INT_GOIT"] = 0
TREND0.loc[(TREND0.INT_THY == 2) & (TREND0.INT_NOD.isnull()), "INT_NOD"] = 0
TREND0.loc[(TREND0.INT_THY == 2) & (TREND0.INT_OTHY.isnull()), "INT_OTHY"] = 0
TREND0.loc[(TREND0.INT_THY == 2) & (TREND0.INT_DNK.isnull()), "INT_DNK"] = 0
TREND0.loc[(TREND0.INT_THY == 2) & (TREND0.INT_RTR.isnull()), "INT_RTR"] = 0
TREND0.loc[(TREND0.INT_THY == 2) & (TREND0.INT_RIT.isnull()), "INT_RIT"] = 2

print("before removing preg participants", TREND0.shape)
# removing currently pregnant participants
TREND0 = TREND0.drop(TREND0[TREND0.INT_PREG == 1].index)
print("after removing preg participants", TREND0.shape)


# removing currently_preg and radio_therapy_year features from dataset
TREND0 = TREND0.drop(["INT_PREG"], axis=1)
print("removing preg feature", TREND0.shape)

# standardise binary coding variables
TREND0['SEX'] = TREND0['SEX'].replace(1, 0).replace(2, 1)
TREND0['INT_THY'] = TREND0['INT_THY'].replace(2, 0)
TREND0['INT_THY'] = TREND0['INT_THY'].replace(998, 0)
TREND0['INT_RIT'] = TREND0['INT_RIT'].replace(2, 0)
TREND0['INT_HYPER'] = TREND0['INT_HYPER'].replace(-1, 1)
TREND0['INT_HYPO'] = TREND0['INT_HYPO'].replace(-1, 1)
TREND0['INT_GOIT'] = TREND0['INT_GOIT'].replace(-1, 1)
TREND0['INT_NOD'] = TREND0['INT_NOD'].replace(-1, 1)
TREND0['INT_OTHY'] = TREND0['INT_OTHY'].replace(-1, 1)
TREND0['INT_DNK'] = TREND0['INT_DNK'].replace(-1, 1)
TREND0['INT_RTR'] = TREND0['INT_RTR'].replace(-1, 1)

# Convert not assessed and unlevied to 0
TREND0['THY_HR'] = TREND0['THY_HR'].replace(8, 0).replace(9, 0)
TREND0['THR_NNR'] = TREND0['THR_NNR'].replace(8, 0).replace(9, 0)
TREND0['THY_HL'] = TREND0['THY_HL'].replace(8, 0).replace(9, 0)
TREND0['THR_NNL'] = TREND0['THR_NNL'].replace(8, 0).replace(9, 0)
TREND0['THY_ER'] = TREND0['THY_ER'].replace(8, 0).replace(9, 0)
TREND0['THY_EL'] = TREND0['THY_EL'].replace(8, 0).replace(9, 0)
TREND0['THY_FIN'] = TREND0['THY_FIN'].replace(8, 0).replace(9, 0)

# fill in NA values with 0
TREND0 = TREND0.fillna(0)


# RENAME THYROID FEATURES FOR create_thyroid_classes() function
SHIP1 = SHIP1.rename(columns={"FT4_RE": "FT4"}).rename(columns={"TSH_RE": "TSH"})
SHIP2 = SHIP2.rename(columns={"FT4_RE": "FT4"})
TREND0 = TREND0.rename(columns={"FT4_RE": "FT4"})


def create_thyroid_classes(data, ft4_lr, ft4_ur, tsh_lr, tsh_ur):
    data.loc[data['FT4'] > ft4_ur, 'BIN_HYPER'] = 1
    data.loc[data['FT4'] <= ft4_ur, 'BIN_HYPER'] = 0
    data.loc[data['FT4'] >= ft4_lr, 'BIN_HYPO'] = 0
    data.loc[data['FT4'] < ft4_lr, 'BIN_HYPO'] = 1
    print("Before class groups: ", data.shape)
    hypo_class_data = data[data.TSH >= tsh_lr]
    print("hypo: ", hypo_class_data.shape)
    hyper_class_data = data[data.TSH <= tsh_ur]
    print("hyper: ", hyper_class_data.shape)
    return hypo_class_data, hyper_class_data

print(colored("\nCREATING CLASSIFICATION GROUPS............", "blue"))
print(colored("\nSHIP0", "yellow"))
hypo_SHIP0, hyper_SHIP0 = create_thyroid_classes(data=SHIP0, ft4_lr=12, ft4_ur=22, tsh_lr=0.27, tsh_ur=4.2)
print(hypo_SHIP0['BIN_HYPO'].value_counts())
print(hyper_SHIP0['BIN_HYPER'].value_counts())

print(colored("\nSHIP1", "yellow"))
hypo_SHIP1, hyper_SHIP1 = create_thyroid_classes(data=SHIP1, ft4_lr=12, ft4_ur=22, tsh_lr=0.27, tsh_ur=4.2)
print(hypo_SHIP1['BIN_HYPO'].value_counts())
print(hyper_SHIP1['BIN_HYPER'].value_counts())

print(colored("\nSHIP2", "yellow"))
hypo_SHIP2, hyper_SHIP2 = create_thyroid_classes(data=SHIP2, ft4_lr=12, ft4_ur=22, tsh_lr=0.27, tsh_ur=4.2)
print(hypo_SHIP2['BIN_HYPO'].value_counts())
print(hyper_SHIP2['BIN_HYPER'].value_counts())

print(colored("\nSHIP3", "yellow"))
hypo_SHIP3, hyper_SHIP3 = create_thyroid_classes(data=SHIP3, ft4_lr=12, ft4_ur=22, tsh_lr=0.27, tsh_ur=4.2)
print(hypo_SHIP3['BIN_HYPO'].value_counts())
print(hyper_SHIP3['BIN_HYPER'].value_counts())

print(colored("\nTREND0", "yellow"))
hypo_TREND0, hyper_TREND0 = create_thyroid_classes(data=TREND0, ft4_lr=12, ft4_ur=22, tsh_lr=0.27, tsh_ur=4.2)
print(hypo_TREND0['BIN_HYPO'].value_counts())
print(hyper_TREND0['BIN_HYPER'].value_counts())


def create_reg_data(data):
    print("Before reg group: ", data.shape)
    reg_data = data[data.TSH <= 4.2]
    reg_data = reg_data[reg_data.TSH >= 0.27]
    print("After reg group: ", reg_data.shape)
    return reg_data

print(colored("\nCREATING REGRESSION GROUPS............", "blue"))
print(colored("\nSHIP0", "yellow"))
reg_SHIP0 = create_reg_data(SHIP0)
reg_SHIP0 = reg_SHIP0.drop(["BIN_HYPO", "BIN_HYPER"], axis=1)

print(colored("\nSHIP1", "yellow"))
reg_SHIP1 = create_reg_data(SHIP1)
reg_SHIP1 = reg_SHIP1.drop(["BIN_HYPO", "BIN_HYPER"], axis=1)

print(colored("\nSHIP2", "yellow"))
reg_SHIP2 = create_reg_data(SHIP2)
reg_SHIP2 = reg_SHIP2.drop(["BIN_HYPO", "BIN_HYPER"], axis=1)

print(colored("\nSHIP3", "yellow"))
reg_SHIP3 = create_reg_data(SHIP3)
reg_SHIP3 = reg_SHIP3.drop(["BIN_HYPO", "BIN_HYPER"], axis=1)

print(colored("\nTREND0", "yellow"))
reg_TREND0 = create_reg_data(TREND0)
reg_TREND0 = reg_TREND0.drop(["BIN_HYPO", "BIN_HYPER"], axis=1)

# CREATING SCALED DATAFRAMES OF ALL DATASETS
def data_scaler(data, labels):
    scale_col_names = labels.scale.tolist()
    scale_col_list = [scale_col_names for scale_col_names in scale_col_names if
                       str(scale_col_names) != 'nan']
    scaler = MinMaxScaler()
    data[scale_col_list] = scaler.fit_transform(data[scale_col_list])
    return data


hypo_SHIP0_scaled = data_scaler(hypo_SHIP0, SHIP0_labels)
hypo_SHIP1_scaled = data_scaler(hypo_SHIP1, SHIP1_labels)
hypo_SHIP2_scaled = data_scaler(hypo_SHIP2, SHIP2_labels)
hypo_SHIP3_scaled = data_scaler(hypo_SHIP3, SHIP3_labels)
hypo_TREND0_scaled = data_scaler(hypo_TREND0, TREND0_labels)

hyper_SHIP0_scaled = data_scaler(hyper_SHIP0, SHIP0_labels)
hyper_SHIP1_scaled = data_scaler(hyper_SHIP1, SHIP1_labels)
hyper_SHIP2_scaled = data_scaler(hyper_SHIP2, SHIP2_labels)
hyper_SHIP3_scaled = data_scaler(hyper_SHIP3, SHIP3_labels)
hyper_TREND0_scaled = data_scaler(hyper_TREND0, TREND0_labels)

reg_SHIP0_scaled = data_scaler(reg_SHIP0, SHIP0_labels)
reg_SHIP1_scaled = data_scaler(reg_SHIP1, SHIP1_labels)
reg_SHIP2_scaled = data_scaler(reg_SHIP2, SHIP2_labels)
reg_SHIP3_scaled = data_scaler(reg_SHIP3, SHIP3_labels)
reg_TREND0_scaled = data_scaler(reg_TREND0, TREND0_labels)









