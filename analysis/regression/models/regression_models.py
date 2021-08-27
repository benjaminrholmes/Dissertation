# IMPORTS ###############################################
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from termcolor import colored
from pre_processing.pre_processing import reg_SHIP0_scaled, reg_SHIP1_scaled, reg_SHIP2_scaled, reg_SHIP3_scaled, \
    reg_TREND0_scaled



print(colored("\n...........REGRESSION MODELLING STARTED............", "green"))


def data_split(data, test_size):
    X, y = data.drop(['FT4'], axis=1), data[['FT4']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)
    return X, y, X_train, X_test, y_train, y_test


X0, y0, X0_train, X0_test, y0_train, y0_test = data_split(reg_SHIP0_scaled, 0.3)
X1, y1, X1_train, X1_test, y1_train, y1_test = data_split(reg_SHIP1_scaled, 0.3)
X2, y2, X2_train, X2_test, y2_train, y2_test = data_split(reg_SHIP2_scaled, 0.3)
X3, y3, X3_train, X3_test, y3_train, y3_test = data_split(reg_SHIP3_scaled, 0.3)
XT, yT, XT_train, XT_test, yT_train, yT_test = data_split(reg_TREND0_scaled, 0.3)

print(colored("\nSHIP0 XGBOOST............\n\n", "blue"))

# Create the parameter grid: gbm_param_grid
S0_gbm_param_grid = {
    'objective': ["reg:squarederror"],
    'n_estimators': [998],
    'colsample_bytree': [0.25],
    'learning_rate': [0.004],
    'subsample': [0.25],
    'max_depth': [2],
    'min_child_weight': [8]
}

# Instantiate the regressor model for grid search
S0_gbm = xgb.XGBRegressor(seed=123)

# K fold cross validation
cv = KFold(n_splits=4)

# Perform grid search: grid_mse
S0_grid_mse = GridSearchCV(estimator=S0_gbm, param_grid=S0_gbm_param_grid,
                           scoring='neg_mean_squared_error', cv=cv, verbose=1)
# Fit the GridsearchCV
S0_grid_mse.fit(X0_train, y0_train)

# Best parameter combination
S0_params = S0_grid_mse.best_params_

# Print the best parameters and lowest RMSE
print("Best parameters found: ", S0_params)
print("Lowest RMSE found: ", np.sqrt(np.abs(S0_grid_mse.best_score_)))

S0_best_max_depth = S0_grid_mse.best_params_['max_depth']
S0_best_min_child_weight = S0_grid_mse.best_params_['min_child_weight']
S0_best_n_estimators = S0_grid_mse.best_params_['n_estimators']
S0_best_learning_rate = S0_grid_mse.best_params_['learning_rate']
S0_best_colsample_bytree = S0_grid_mse.best_params_['colsample_bytree']
S0_best_subsample = S0_grid_mse.best_params_['subsample']

# Create the regressor model updated with the best parameters
S0_xgb_mod = xgb.XGBRegressor(objective="reg:squarederror",
                              learning_rate=S0_best_learning_rate,
                              n_estimators=S0_best_n_estimators,
                              subsample=S0_best_subsample,
                              min_child_weight=S0_best_min_child_weight,
                              max_depth=S0_best_max_depth,
                              colsample_bytree=S0_best_colsample_bytree,
                              seed=123)

# Fit the model to the training set
S0_xgb_mod.fit(X0_train, y0_train)

# Prediction of the test set: y_preds
S0_xgb_train_preds = S0_xgb_mod.predict(X0_train)
S0_xgb_test_preds = S0_xgb_mod.predict(X0_test)

S0_xgb_RMSE_train = mean_squared_error(y0_train, S0_xgb_train_preds, squared=False)
S0_xgb_RMSE_test = mean_squared_error(y0_test, S0_xgb_test_preds, squared=False)
S0_xgb_R2_train = r2_score(y0_train, S0_xgb_train_preds)
S0_xgb_R2_test = r2_score(y0_test, S0_xgb_test_preds)
S0_xgb_MAE_train = mean_absolute_error(y0_train, S0_xgb_train_preds)
S0_xgb_MAE_test = mean_absolute_error(y0_test, S0_xgb_test_preds)

print(colored("\nSHIP0 SVM............\n\n", "blue"))

S0_svr_mod = svm.SVR(C=0.005)

S0_svr_mod.fit(X0_train, y0_train.values.ravel())

# Prediction of the test set: y_preds
S0_svr_train_preds = S0_svr_mod.predict(X0_train)
S0_svr_test_preds = S0_svr_mod.predict(X0_test)

S0_svr_RMSE_train = mean_squared_error(y0_train, S0_svr_train_preds, squared=False)
S0_svr_RMSE_test = mean_squared_error(y0_test, S0_svr_test_preds, squared=False)
S0_svr_R2_train = r2_score(y0_train, S0_svr_train_preds)
S0_svr_R2_test = r2_score(y0_test, S0_svr_test_preds)
S0_svr_MAE_train = mean_absolute_error(y0_train, S0_svr_train_preds)
S0_svr_MAE_test = mean_absolute_error(y0_test, S0_svr_test_preds)

print(colored("\nSHIP0 MLP............\n\n", "blue"))

S0_nn_mod = MLPRegressor(random_state=123,
                         max_iter=500,
                         hidden_layer_sizes=(50, 50, 50),
                         alpha=0.005,
                         learning_rate_init=0.001,
                         activation="logistic").fit(X0_train, y0_train.values.ravel())

S0_nn_mod.predict(X0_test)

S0_nn_train_preds = S0_nn_mod.predict(X0_train)
S0_nn_test_preds = S0_nn_mod.predict(X0_test)

S0_nn_RMSE_train = mean_squared_error(y0_train, S0_nn_train_preds, squared=False)
S0_nn_RMSE_test = mean_squared_error(y0_test, S0_nn_test_preds, squared=False)
S0_nn_R2_train = r2_score(y0_train, S0_nn_train_preds)
S0_nn_R2_test = r2_score(y0_test, S0_nn_test_preds)
S0_nn_MAE_train = mean_absolute_error(y0_train, S0_nn_train_preds)
S0_nn_MAE_test = mean_absolute_error(y0_test, S0_nn_test_preds)

print(colored("\nSHIP1 XGBOOST............\n\n", "blue"))

# Create the parameter grid: gbm_param_grid
S1_gbm_param_grid = {
    'objective': ["reg:squarederror"],
    'n_estimators': [998],
    'colsample_bytree': [0.25],
    'learning_rate': [0.004],
    'subsample': [0.25],
    'max_depth': [2],
    'min_child_weight': [8]
}

# Instantiate the regressor model for grid search
S1_gbm = xgb.XGBRegressor(seed=123)

# K fold cross validation
cv = KFold(n_splits=4)

# Perform grid search: grid_mse
S1_grid_mse = GridSearchCV(estimator=S1_gbm, param_grid=S1_gbm_param_grid,
                           scoring='neg_mean_squared_error', cv=cv, verbose=1)
# Fit the GridsearchCV
S1_grid_mse.fit(X1_train, y1_train)

# Best parameter combination
S1_params = S1_grid_mse.best_params_

# Print the best parameters and lowest RMSE
print("Best parameters found: ", S1_params)
print("Lowest RMSE found: ", np.sqrt(np.abs(S1_grid_mse.best_score_)))

S1_best_max_depth = S1_grid_mse.best_params_['max_depth']
S1_best_min_child_weight = S1_grid_mse.best_params_['min_child_weight']
S1_best_n_estimators = S1_grid_mse.best_params_['n_estimators']
S1_best_learning_rate = S1_grid_mse.best_params_['learning_rate']
S1_best_colsample_bytree = S1_grid_mse.best_params_['colsample_bytree']
S1_best_subsample = S1_grid_mse.best_params_['subsample']

# Create the regressor model updated with the best parameters
S1_xgb_mod = xgb.XGBRegressor(objective="reg:squarederror",
                              learning_rate=S1_best_learning_rate,
                              n_estimators=S1_best_n_estimators,
                              subsample=S1_best_subsample,
                              min_child_weight=S1_best_min_child_weight,
                              max_depth=S1_best_max_depth,
                              colsample_bytree=S1_best_colsample_bytree,
                              seed=123)

# Fit the model to the training set
S1_xgb_mod.fit(X1_train, y1_train)

# Prediction of the test set: y_preds
S1_xgb_train_preds = S1_xgb_mod.predict(X1_train)
S1_xgb_test_preds = S1_xgb_mod.predict(X1_test)

S1_xgb_RMSE_train = mean_squared_error(y1_train, S1_xgb_train_preds, squared=False)
S1_xgb_RMSE_test = mean_squared_error(y1_test, S1_xgb_test_preds, squared=False)
S1_xgb_R2_train = r2_score(y1_train, S1_xgb_train_preds)
S1_xgb_R2_test = r2_score(y1_test, S1_xgb_test_preds)
S1_xgb_MAE_train = mean_absolute_error(y1_train, S1_xgb_train_preds)
S1_xgb_MAE_test = mean_absolute_error(y1_test, S1_xgb_test_preds)

print(colored("\nSHIP1 SVM............\n\n", "blue"))

S1_svr_mod = svm.SVR(kernel="linear")

S1_svr_mod.fit(X1_train, y1_train.values.ravel())

# Prediction of the test set: y_preds
S1_svr_train_preds = S1_svr_mod.predict(X1_train)
S1_svr_test_preds = S1_svr_mod.predict(X1_test)

S1_svr_RMSE_train = mean_squared_error(y1_train, S1_svr_train_preds, squared=False)
S1_svr_RMSE_test = mean_squared_error(y1_test, S1_svr_test_preds, squared=False)
S1_svr_R2_train = r2_score(y1_train, S1_svr_train_preds)
S1_svr_R2_test = r2_score(y1_test, S1_svr_test_preds)
S1_svr_MAE_train = mean_absolute_error(y1_train, S1_svr_train_preds)
S1_svr_MAE_test = mean_absolute_error(y1_test, S1_svr_test_preds)

print(colored("\nSHIP1 MLP............\n\n", "blue"))

S1_nn_mod = MLPRegressor(random_state=123,
                         max_iter=600,
                         hidden_layer_sizes=(100, 100, 100),
                         learning_rate_init=0.001,
                         activation="logistic").fit(X1_train, y1_train.values.ravel())

S1_nn_mod.predict(X1_test)

S1_nn_train_preds = S1_nn_mod.predict(X1_train)
S1_nn_test_preds = S1_nn_mod.predict(X1_test)

S1_nn_RMSE_train = mean_squared_error(y1_train, S1_nn_train_preds, squared=False)
S1_nn_RMSE_test = mean_squared_error(y1_test, S1_nn_test_preds, squared=False)
S1_nn_R2_train = r2_score(y1_train, S1_nn_train_preds)
S1_nn_R2_test = r2_score(y1_test, S1_nn_test_preds)
S1_nn_MAE_train = mean_absolute_error(y1_train, S1_nn_train_preds)
S1_nn_MAE_test = mean_absolute_error(y1_test, S1_nn_test_preds)

print(colored("\nSHIP2 XGBOOST............\n\n", "blue"))

# Create the parameter grid: gbm_param_grid
S2_gbm_param_grid = {
    'objective': ["reg:squarederror"],
    'n_estimators': [998],
    'colsample_bytree': [0.5],
    'learning_rate': [0.004],
    'subsample': [0.5],
    'max_depth': [3],
    'min_child_weight': [2]
}

# Instantiate the regressor model for grid search
S2_gbm = xgb.XGBRegressor(seed=123)

# K fold cross validation
cv = KFold(n_splits=4)

# Perform grid search: grid_mse
S2_grid_mse = GridSearchCV(estimator=S2_gbm, param_grid=S2_gbm_param_grid,
                           scoring='neg_mean_squared_error', cv=cv, verbose=1)
# Fit the GridsearchCV
S2_grid_mse.fit(X2_train, y2_train)

# Best parameter combination
S2_params = S2_grid_mse.best_params_

# Print the best parameters and lowest RMSE
print("Best parameters found: ", S2_params)
print("Lowest RMSE found: ", np.sqrt(np.abs(S2_grid_mse.best_score_)))

S2_best_max_depth = S2_grid_mse.best_params_['max_depth']
S2_best_min_child_weight = S2_grid_mse.best_params_['min_child_weight']
S2_best_n_estimators = S2_grid_mse.best_params_['n_estimators']
S2_best_learning_rate = S2_grid_mse.best_params_['learning_rate']
S2_best_colsample_bytree = S2_grid_mse.best_params_['colsample_bytree']
S2_best_subsample = S2_grid_mse.best_params_['subsample']

# Create the regressor model updated with the best parameters
S2_xgb_mod = xgb.XGBRegressor(objective="reg:squarederror",
                              learning_rate=S2_best_learning_rate,
                              n_estimators=S2_best_n_estimators,
                              subsample=S2_best_subsample,
                              min_child_weight=S2_best_min_child_weight,
                              max_depth=S2_best_max_depth,
                              colsample_bytree=S2_best_colsample_bytree,
                              seed=123)

# Fit the model to the training set
S2_xgb_mod.fit(X2_train, y2_train)

# Prediction of the test set: y_preds
S2_xgb_train_preds = S2_xgb_mod.predict(X2_train)
S2_xgb_test_preds = S2_xgb_mod.predict(X2_test)

S2_xgb_RMSE_train = mean_squared_error(y2_train, S2_xgb_train_preds, squared=False)
S2_xgb_RMSE_test = mean_squared_error(y2_test, S2_xgb_test_preds, squared=False)
S2_xgb_R2_train = r2_score(y2_train, S2_xgb_train_preds)
S2_xgb_R2_test = r2_score(y2_test, S2_xgb_test_preds)
S2_xgb_MAE_train = mean_absolute_error(y2_train, S2_xgb_train_preds)
S2_xgb_MAE_test = mean_absolute_error(y2_test, S2_xgb_test_preds)

print(colored("\nSHIP2 SVM............\n\n", "blue"))

S2_svr_mod = svm.SVR(C=0.01)

S2_svr_mod.fit(X2_train, y2_train.values.ravel())

# Prediction of the test set: y_preds
S2_svr_train_preds = S2_svr_mod.predict(X2_train)
S2_svr_test_preds = S2_svr_mod.predict(X2_test)

S2_svr_RMSE_train = mean_squared_error(y2_train, S2_svr_train_preds, squared=False)
S2_svr_RMSE_test = mean_squared_error(y2_test, S2_svr_test_preds, squared=False)
S2_svr_R2_train = r2_score(y2_train, S2_svr_train_preds)
S2_svr_R2_test = r2_score(y2_test, S2_svr_test_preds)
S2_svr_MAE_train = mean_absolute_error(y2_train, S2_svr_train_preds)
S2_svr_MAE_test = mean_absolute_error(y2_test, S2_svr_test_preds)

print(colored("\nSHIP2 MLP............\n\n", "blue"))

S2_nn_mod = MLPRegressor(random_state=123,
                         max_iter=500,
                         hidden_layer_sizes=(50, 50, 50),
                         alpha=0.005,
                         learning_rate_init=0.001,
                         activation="logistic").fit(X2_train, y2_train.values.ravel())

S2_nn_mod.predict(X2_test)

S2_nn_train_preds = S2_nn_mod.predict(X2_train)
S2_nn_test_preds = S2_nn_mod.predict(X2_test)

S2_nn_RMSE_train = mean_squared_error(y2_train, S2_nn_train_preds, squared=False)
S2_nn_RMSE_test = mean_squared_error(y2_test, S2_nn_test_preds, squared=False)
S2_nn_R2_train = r2_score(y2_train, S2_nn_train_preds)
S2_nn_R2_test = r2_score(y2_test, S2_nn_test_preds)
S2_nn_MAE_train = mean_absolute_error(y2_train, S2_nn_train_preds)
S2_nn_MAE_test = mean_absolute_error(y2_test, S2_nn_test_preds)

print(colored("\nSHIP3 XGBOOST............\n\n", "blue"))

# Create the parameter grid: gbm_param_grid
S3_gbm_param_grid = {
    'objective': ["reg:squarederror"],
    'n_estimators': [947],
    'colsample_bytree': [0.3],
    'learning_rate': [0.006],
    'subsample': [0.3],
    'max_depth': [2],
    'min_child_weight': [10]
}

# Instantiate the regressor model for grid search
S3_gbm = xgb.XGBRegressor(seed=123)

# K fold cross validation
cv = KFold(n_splits=4)

# Perform grid search: grid_mse
S3_grid_mse = GridSearchCV(estimator=S3_gbm, param_grid=S3_gbm_param_grid,
                           scoring='neg_mean_squared_error', cv=cv, verbose=1)
# Fit the GridsearchCV
S3_grid_mse.fit(X3_train, y3_train)

# Best parameter combination
S3_params = S3_grid_mse.best_params_

# Print the best parameters and lowest RMSE
print("Best parameters found: ", S3_params)
print("Lowest RMSE found: ", np.sqrt(np.abs(S3_grid_mse.best_score_)))

S3_best_max_depth = S3_grid_mse.best_params_['max_depth']
S3_best_min_child_weight = S3_grid_mse.best_params_['min_child_weight']
S3_best_n_estimators = S3_grid_mse.best_params_['n_estimators']
S3_best_learning_rate = S3_grid_mse.best_params_['learning_rate']
S3_best_colsample_bytree = S3_grid_mse.best_params_['colsample_bytree']
S3_best_subsample = S3_grid_mse.best_params_['subsample']

# Create the regressor model updated with the best parameters
S3_xgb_mod = xgb.XGBRegressor(objective="reg:squarederror",
                              learning_rate=S3_best_learning_rate,
                              n_estimators=S3_best_n_estimators,
                              subsample=S3_best_subsample,
                              min_child_weight=S3_best_min_child_weight,
                              max_depth=S3_best_max_depth,
                              colsample_bytree=S3_best_colsample_bytree,
                              seed=123)

# Fit the model to the training set
S3_xgb_mod.fit(X3_train, y3_train)

# Prediction of the test set: y_preds
S3_xgb_train_preds = S3_xgb_mod.predict(X3_train)
S3_xgb_test_preds = S3_xgb_mod.predict(X3_test)

S3_xgb_RMSE_train = mean_squared_error(y3_train, S3_xgb_train_preds, squared=False)
S3_xgb_RMSE_test = mean_squared_error(y3_test, S3_xgb_test_preds, squared=False)
S3_xgb_R2_train = r2_score(y3_train, S3_xgb_train_preds)
S3_xgb_R2_test = r2_score(y3_test, S3_xgb_test_preds)
S3_xgb_MAE_train = mean_absolute_error(y3_train, S3_xgb_train_preds)
S3_xgb_MAE_test = mean_absolute_error(y3_test, S3_xgb_test_preds)

print(colored("\nSHIP3 SVM............\n\n", "blue"))

S3_svr_mod_best = svm.SVR(kernel="linear")

S3_svr_mod_best.fit(X3_train, y3_train.values.ravel())

# Prediction of the test set: y_preds
S3_svr_train_preds = S3_svr_mod_best.predict(X3_train)
S3_svr_test_preds = S3_svr_mod_best.predict(X3_test)

S3_svr_RMSE_train = mean_squared_error(y3_train, S3_svr_train_preds, squared=False)
S3_svr_RMSE_test = mean_squared_error(y3_test, S3_svr_test_preds, squared=False)
S3_svr_R2_train = r2_score(y3_train, S3_svr_train_preds)
S3_svr_R2_test = r2_score(y3_test, S3_svr_test_preds)
S3_svr_MAE_train = mean_absolute_error(y3_train, S3_svr_train_preds)
S3_svr_MAE_test = mean_absolute_error(y3_test, S3_svr_test_preds)

print(colored("\nSHIP3 MLP............\n\n", "blue"))

S3_nn_mod = MLPRegressor(random_state=123,
                         max_iter=500,
                         hidden_layer_sizes=(50, 50, 50),
                         alpha=0.005,
                         learning_rate_init=0.001,
                         activation="logistic").fit(X3_train, y3_train.values.ravel())

S3_nn_mod.predict(X3_test)

S3_nn_train_preds = S3_nn_mod.predict(X3_train)
S3_nn_test_preds = S3_nn_mod.predict(X3_test)

S3_nn_RMSE_train = mean_squared_error(y3_train, S3_nn_train_preds, squared=False)
S3_nn_RMSE_test = mean_squared_error(y3_test, S3_nn_test_preds, squared=False)
S3_nn_R2_train = r2_score(y3_train, S3_nn_train_preds)
S3_nn_R2_test = r2_score(y3_test, S3_nn_test_preds)
S3_nn_MAE_train = mean_absolute_error(y3_train, S3_nn_train_preds)
S3_nn_MAE_test = mean_absolute_error(y3_test, S3_nn_test_preds)

print(colored("\nTREND0 XGBOOST............\n\n", "blue"))

# Create the parameter grid: gbm_param_grid
T0_gbm_param_grid = {
    'objective': ["reg:squarederror"],
    'n_estimators': [500],
    'colsample_bytree': [0.25],
    'learning_rate': [0.015],
    'subsample': [0.25],
    'max_depth': [2],
    'min_child_weight': [10]
}

# Instantiate the regressor model for grid search
T0_gbm = xgb.XGBRegressor(seed=123)

# K fold cross validation
cv = KFold(n_splits=4)

# Perform grid search: grid_mse
T0_grid_mse = GridSearchCV(estimator=T0_gbm, param_grid=T0_gbm_param_grid,
                           scoring='neg_mean_squared_error', cv=cv, verbose=1)
# Fit the GridsearchCV
T0_grid_mse.fit(XT_train, yT_train)

# Best parameter combination
T0_params = T0_grid_mse.best_params_

# Print the best parameters and lowest RMSE
print("Best parameters found: ", T0_params)
print("Lowest RMSE found: ", np.sqrt(np.abs(T0_grid_mse.best_score_)))

T0_best_max_depth = T0_grid_mse.best_params_['max_depth']
T0_best_min_child_weight = T0_grid_mse.best_params_['min_child_weight']
T0_best_n_estimators = T0_grid_mse.best_params_['n_estimators']
T0_best_learning_rate = T0_grid_mse.best_params_['learning_rate']
T0_best_colsample_bytree = T0_grid_mse.best_params_['colsample_bytree']
T0_best_subsample = T0_grid_mse.best_params_['subsample']

# Create the regressor model updated with the best parameters
T0_xgb_mod = xgb.XGBRegressor(objective="reg:squarederror",
                              learning_rate=T0_best_learning_rate,
                              n_estimators=T0_best_n_estimators,
                              subsample=T0_best_subsample,
                              min_child_weight=T0_best_min_child_weight,
                              max_depth=T0_best_max_depth,
                              colsample_bytree=T0_best_colsample_bytree,
                              seed=123)

# Fit the model to the training set
T0_xgb_mod.fit(XT_train, yT_train)

# Prediction of the test set: y_preds
T0_xgb_train_preds = T0_xgb_mod.predict(XT_train)
T0_xgb_test_preds = T0_xgb_mod.predict(XT_test)

T0_xgb_RMSE_train = mean_squared_error(yT_train, T0_xgb_train_preds, squared=False)
T0_xgb_RMSE_test = mean_squared_error(yT_test, T0_xgb_test_preds, squared=False)
T0_xgb_R2_train = r2_score(yT_train, T0_xgb_train_preds)
T0_xgb_R2_test = r2_score(yT_test, T0_xgb_test_preds)
T0_xgb_MAE_train = mean_absolute_error(yT_train, T0_xgb_train_preds)
T0_xgb_MAE_test = mean_absolute_error(yT_test, T0_xgb_test_preds)

print(colored("\nTREND0 SVM............\n\n", "blue"))

T0_svr_mod_best = svm.SVR(kernel="linear")

T0_svr_mod_best.fit(XT_train, yT_train.values.ravel())

# Prediction of the test set: y_preds
T0_svr_train_preds = T0_svr_mod_best.predict(XT_train)
T0_svr_test_preds = T0_svr_mod_best.predict(XT_test)

T0_svr_RMSE_train = mean_squared_error(yT_train, T0_svr_train_preds, squared=False)
T0_svr_RMSE_test = mean_squared_error(yT_test, T0_svr_test_preds, squared=False)
T0_svr_R2_train = r2_score(yT_train, T0_svr_train_preds)
T0_svr_R2_test = r2_score(yT_test, T0_svr_test_preds)
T0_svr_MAE_train = mean_absolute_error(yT_train, T0_svr_train_preds)
T0_svr_MAE_test = mean_absolute_error(yT_test, T0_svr_test_preds)

print(colored("\nTREND0 MLP............\n\n", "blue"))

T0_nn_mod = MLPRegressor(random_state=123,
                         max_iter=500,
                         hidden_layer_sizes=(50, 50, 50),
                         alpha=0.005,
                         learning_rate_init=0.001,
                         activation="logistic").fit(XT_train, yT_train.values.ravel())

T0_nn_mod.predict(XT_test)

T0_nn_train_preds = T0_nn_mod.predict(XT_train)
T0_nn_test_preds = T0_nn_mod.predict(XT_test)

T0_nn_RMSE_train = mean_squared_error(yT_train, T0_nn_train_preds, squared=False)
T0_nn_RMSE_test = mean_squared_error(yT_test, T0_nn_test_preds, squared=False)
T0_nn_R2_train = r2_score(yT_train, T0_nn_train_preds)
T0_nn_R2_test = r2_score(yT_test, T0_nn_test_preds)
T0_nn_MAE_train = mean_absolute_error(yT_train, T0_nn_train_preds)
T0_nn_MAE_test = mean_absolute_error(yT_test, T0_nn_test_preds)

print(colored("\nREGRESSION RESULTS............", "blue"))

# initialise data of lists.
results = {'SHIP-0 XGBoost Train': [S0_xgb_RMSE_train, S0_xgb_R2_train, S0_xgb_MAE_train],
           'SHIP-0 XGBoost Test': [S0_xgb_RMSE_test, S0_xgb_R2_test, S0_xgb_MAE_test],
           'SHIP-0 SVM Train': [S0_svr_RMSE_train, S0_svr_R2_train, S0_svr_MAE_train],
           'SHIP-0 SVM Test': [S0_svr_RMSE_test, S0_svr_R2_test, S0_svr_MAE_test],
           'SHIP-0 MLP Train': [S0_nn_RMSE_train, S0_nn_R2_train, S0_nn_MAE_train],
           'SHIP-0 MLP Test': [S0_nn_RMSE_test, S0_nn_R2_test, S0_nn_MAE_test],
           'SHIP-1 XGBoost Train': [S1_xgb_RMSE_train, S1_xgb_R2_train, S1_xgb_MAE_train],
           'SHIP-1 XGBoost Test': [S1_xgb_RMSE_test, S1_xgb_R2_test, S1_xgb_MAE_test],
           'SHIP-1 SVM Train': [S1_svr_RMSE_train, S1_svr_R2_train, S1_svr_MAE_train],
           'SHIP-1 SVM Test': [S1_svr_RMSE_test, S1_svr_R2_test, S1_svr_MAE_test],
           'SHIP-1 MLP Train': [S1_nn_RMSE_train, S1_nn_R2_train, S1_nn_MAE_train],
           'SHIP-1 MLP Test': [S1_nn_RMSE_test, S1_nn_R2_test, S1_nn_MAE_test],
           'SHIP-2 XGBoost Train': [S2_xgb_RMSE_train, S2_xgb_R2_train, S2_xgb_MAE_train],
           'SHIP-2 XGBoost Test': [S2_xgb_RMSE_test, S2_xgb_R2_test, S2_xgb_MAE_test],
           'SHIP-2 SVM Train': [S2_svr_RMSE_train, S2_svr_R2_train, S2_svr_MAE_train],
           'SHIP-2 SVM Test': [S2_svr_RMSE_test, S2_svr_R2_test, S2_svr_MAE_test],
           'SHIP-2 MLP Train': [S2_nn_RMSE_train, S2_nn_R2_train, S2_nn_MAE_train],
           'SHIP-2 MLP Test': [S2_nn_RMSE_test, S2_nn_R2_test, S2_nn_MAE_test],
           'SHIP-3 XGBoost Train': [S3_xgb_RMSE_train, S3_xgb_R2_train, S3_xgb_MAE_train],
           'SHIP-3 XGBoost Test': [S3_xgb_RMSE_test, S3_xgb_R2_test, S3_xgb_MAE_test],
           'SHIP-3 SVM Train': [S3_svr_RMSE_train, S3_svr_R2_train, S3_svr_MAE_train],
           'SHIP-3 SVM Test': [S3_svr_RMSE_test, S3_svr_R2_test, S3_svr_MAE_test],
           'SHIP-3 MLP Train': [S3_nn_RMSE_train, S3_nn_R2_train, S3_nn_MAE_train],
           'SHIP-3 MLP Test': [S3_nn_RMSE_test, S3_nn_R2_test, S3_nn_MAE_test],
           'TREND-0 XGBoost Train': [T0_xgb_RMSE_train, T0_xgb_R2_train, T0_xgb_MAE_train],
           'TREND-0 XGBoost Test': [T0_xgb_RMSE_test, T0_xgb_R2_test, T0_xgb_MAE_test],
           'TREND-0 SVM Train': [T0_svr_RMSE_train, T0_svr_R2_train, T0_svr_MAE_train],
           'TREND-0 SVM Test': [T0_svr_RMSE_test, T0_svr_R2_test, T0_svr_MAE_test],
           'TREND-0 MLP Train': [T0_nn_RMSE_train, T0_nn_R2_train, T0_nn_MAE_train],
           'TREND-0 MLP Test': [T0_nn_RMSE_test, T0_nn_R2_test, T0_nn_MAE_test]
           }

# Creates pandas DataFrame.
results_df = pd.DataFrame(results, index=['RMSE',
                                          'R2',
                                          'MAE'])

print(results_df)

S0_target_std = y0.std()
S0_target_std = S0_target_std.at["FT4"]
S1_target_std = y1.std()
S1_target_std = S1_target_std.at["FT4"]
S2_target_std = y2.std()
S2_target_std = S2_target_std.at["FT4"]
S3_target_std = y3.std()
S3_target_std = S3_target_std.at["FT4"]
T0_target_std = yT.std()
T0_target_std = T0_target_std.at["FT4"]


print(colored("\nTARGET STANDARD DEVIATIONS............", "blue"))
print("\nS0 target std:", S0_target_std)
print("\nS1 target std:", S1_target_std)
print("\nS2 target std:", S2_target_std)
print("\nS3 target std:", S3_target_std)
print("\nT0 target std:", T0_target_std)

print(colored("\n...........REGRESSION MODELLING FINISHED............", "green"))


results_df.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\regression_results.csv")