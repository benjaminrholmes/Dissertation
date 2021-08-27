import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from pre_processing.pre_processing import hyper_SHIP1_scaled
from termcolor import colored

print(colored("\n...........SHIP1 HYPERTHYROIDISM CLASSIFICATION XGBOOST MODELLING STARTED............\n\n", "green"))

X, y = hyper_SHIP1_scaled.drop(['FT4', 'BIN_HYPO', 'BIN_HYPER'], axis=1), hyper_SHIP1_scaled[['BIN_HYPER']]

smote = SMOTE(sampling_strategy="minority")
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'objective': ["binary:logistic"],
    'n_estimators': [998],
    'colsample_bytree': [0.25],
    'learning_rate': [0.004],
    'subsample': [0.25],
    'max_depth': [2],
    'min_child_weight': [8],
    'eval_metric': ["logloss"]
}

# Instantiate the classifier model for grid search
gbm = xgb.XGBClassifier(seed=123, use_label_encoder=False)

# K fold cross validation
cv = StratifiedKFold(n_splits=4)

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring='roc_auc', cv=cv, verbose=1)
# Fit the GridsearchCV
grid_mse.fit(X_train, y_train)

# Best parameter combination
params = grid_mse.best_params_

# Print the best parameters and lowest RMSE
print("Best parameters found: ", params)
print("Lowest ROC AUC found: ", np.sqrt(np.abs(grid_mse.best_score_)))

best_max_depth = grid_mse.best_params_['max_depth']
best_min_child_weight = grid_mse.best_params_['min_child_weight']
best_n_estimators = grid_mse.best_params_['n_estimators']
best_learning_rate = grid_mse.best_params_['learning_rate']
best_colsample_bytree = grid_mse.best_params_['colsample_bytree']
best_subsample = grid_mse.best_params_['subsample']

# Create the DMatrix: housing_dmatrix
dmatrix = xgb.DMatrix(data=X, label=y)
# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=3, num_boost_round=999, early_stopping_rounds=10,
                    metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Create the regressor model updated with the best parameters
mod_best = xgb.XGBClassifier(objective="binary:logistic",
                             learning_rate=best_learning_rate,
                             n_estimators=best_n_estimators,
                             subsample=best_subsample,
                             min_child_weight=best_min_child_weight,
                             max_depth=best_max_depth,
                             colsample_bytree=best_colsample_bytree,
                             seed=123, use_label_encoder=False)

# Fit the model to the training set
mod_best.fit(X_train, y_train)

# Prediction of the test set: y_preds
xgb_train_preds = mod_best.predict(X_train)
xgb_test_preds = mod_best.predict(X_test)

cm = confusion_matrix(y_test, xgb_test_preds)
cm2 = confusion_matrix(y_test, xgb_test_preds, normalize="true")

accuracy_train = accuracy_score(y_train, xgb_train_preds)
precision_train = precision_score(y_train, xgb_train_preds)
recall_train = recall_score(y_train, xgb_train_preds)
f1_score_train = f1_score(y_train, xgb_train_preds)
auc_train = roc_auc_score(y_train, xgb_train_preds)
log_loss_train = log_loss(y_train, xgb_train_preds)

accuracy_test = accuracy_score(y_test, xgb_test_preds)
precision_test = precision_score(y_test, xgb_test_preds)
recall_test = recall_score(y_test, xgb_test_preds)
f1_score_test = f1_score(y_test, xgb_test_preds)
auc_test = roc_auc_score(y_test, xgb_test_preds)
log_loss_test = log_loss(y_test, xgb_test_preds)

print("\nModel Report")
print("\n Confusion Matrix")
print(cm)
print("\n Normalized Confusion Matrix")
print(cm2)

# initialise data of lists.
results = {'Train': [accuracy_train, precision_train, recall_train, f1_score_train, auc_train, log_loss_train],
           'Test': [accuracy_test, precision_test, recall_test, f1_score_test, auc_test, log_loss_test]}

# Creates pandas DataFrame.
results_df = pd.DataFrame(results, index=['Accuracy',
                                          'Precision',
                                          'Recall',
                                          'F1',
                                          'ROC AUC',
                                          'log loss'])

# print the data
print(results_df)

results_df.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\classification\\result_data\\S1\\S1_XGB_hyper.csv")

xgb_y_pred_per = mod_best.predict_proba(X_test)
xgb_mod_per = mod_best
xgb_y_per_test = y_test
xgb_X_per_test = X_test

print(colored("\n...........SHIP1 HYPERTHYROIDISM CLASSIFICATION XGBOOST MODELLING FINISHED............\n\n", "green"))
