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
TEXT
___