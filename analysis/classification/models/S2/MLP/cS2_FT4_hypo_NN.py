import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pre_processing.pre_processing import hypo_SHIP2_scaled
from termcolor import colored

print(colored("\n...........SHIP2 HYPOTHYROIDISM CLASSIFICATION MLP MODELLING STARTED............\n\n", "green"))


X, y = hypo_SHIP2_scaled.drop(['FT4', 'BIN_HYPO', 'BIN_HYPER'], axis=1), hypo_SHIP2_scaled[['BIN_HYPO']]

smote = SMOTE(sampling_strategy="minority")
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

mod_best = MLPClassifier(random_state=123,
                         max_iter=1500,
                         hidden_layer_sizes=(100, 100, 100),
                         alpha=0.0008,
                         learning_rate_init=0.001,
                         activation="logistic").fit(X_train, y_train.values.ravel())

mod_best.predict(X_test)

# Fit the model to the training set
mod_best.fit(X_train, y_train.values.ravel())

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

results_df.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\classification\\result_data\\S2\\S2_MLP_hypo.csv")

nn_y_pred_po = mod_best.predict_proba(X_test)
nn_mod_po = mod_best
nn_y_po_test = y_test
nn_X_po_test = X_test

print(colored("\n...........SHIP2 HYPOTHYROIDISM CLASSIFICATION MLP MODELLING FINISHED............\n\n", "green"))
