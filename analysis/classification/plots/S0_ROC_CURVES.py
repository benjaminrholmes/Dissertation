import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix
import shap
import pandas as pd
import numpy as np
from termcolor import colored

from analysis.classification.models.S0.MLP.cS0_FT4_hyper_NN import nn_y_per_test, nn_y_pred_per, nn_mod_per, nn_X_per_test
from analysis.classification.models.S0.MLP.cS0_FT4_hypo_NN import nn_y_po_test, nn_y_pred_po, nn_mod_po, nn_X_po_test
from analysis.classification.models.S0.SVM.cS0_FT4_hyper_SVM import svm_y_per_test, svm_y_pred_per, svm_mod_per, svm_X_per_test
from analysis.classification.models.S0.SVM.cS0_FT4_hypo_SVM import svm_y_po_test, svm_y_pred_po, svm_mod_po, svm_X_po_test
from analysis.classification.models.S0.XGBoost.cS0_FT4_hyper_XGB import xgb_y_per_test, xgb_y_pred_per, xgb_mod_per, xgb_X_per_test
from analysis.classification.models.S0.XGBoost.cS0_FT4_hypo_XGB import xgb_y_po_test, xgb_y_pred_po, xgb_mod_po, xgb_X_po_test

print(colored("\n...........SHIP0 CLASSIFICATION PLOTTING STARTED............", "green"))

# XGBoost
xgb_po_fpr, xgb_po_tpr, _ = roc_curve(xgb_y_po_test.values, xgb_y_pred_po[:,1])
xgb_po_roc_auc = auc(xgb_po_fpr, xgb_po_tpr)
xgb_per_fpr, xgb_per_tpr, _ = roc_curve(xgb_y_per_test.values, xgb_y_pred_per[:,1])
xgb_per_roc_auc = auc(xgb_per_fpr, xgb_per_tpr)

#SVM
svm_po_fpr, svm_po_tpr, _ = roc_curve(svm_y_po_test.values, svm_y_pred_po[:,1])
svm_po_roc_auc = auc(svm_po_fpr, svm_po_tpr)
svm_per_fpr, svm_per_tpr, _ = roc_curve(svm_y_per_test.values, svm_y_pred_per[:,1])
svm_per_roc_auc = auc(svm_per_fpr, svm_per_tpr)

#NN
nn_po_fpr, nn_po_tpr, _ = roc_curve(nn_y_po_test.values, nn_y_pred_po[:,1])
nn_po_roc_auc = auc(nn_po_fpr, nn_po_tpr)
nn_per_fpr, nn_per_tpr, _ = roc_curve(nn_y_per_test.values, nn_y_pred_per[:,1])
nn_per_roc_auc = auc(nn_per_fpr, nn_per_tpr)


plt.figure()
lw = 1
plt.plot(xgb_per_fpr, xgb_per_tpr, color='#fb00ff',
         lw=lw, linestyle='--', label='XGB Hyperthyroidism (area = %0.2f)' % xgb_per_roc_auc)
plt.plot(xgb_po_fpr, xgb_po_tpr, color='#fb00ff',
         lw=lw, label='XGB Hypothyroidism (area = %0.2f)' % xgb_po_roc_auc)
plt.plot(svm_per_fpr, svm_per_tpr, color='#0080ff',
         lw=lw, linestyle='--', label='SVM Hyperthyroidism (area = %0.2f)' % svm_per_roc_auc)
plt.plot(svm_po_fpr, svm_po_tpr, color='#0080ff',
         lw=lw, label='SVM Hypothyroidism (area = %0.2f)' % svm_po_roc_auc)
plt.plot(nn_per_fpr, nn_per_tpr, color='#00ff2f',
         lw=lw, linestyle='--', label='MLP Hyperthyroidism (area = %0.2f)' % nn_per_roc_auc)
plt.plot(nn_po_fpr, nn_po_tpr, color='#00ff2f',
         lw=lw, label='MLP Hypothyroidism (area = %0.2f)' % nn_po_roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, color="black", linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SHIP-0 ROC curve')
plt.legend(loc="lower right")
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\classification\\result_data\\S0\\S0_ROC.png')
plt.show()

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 15))

plot_confusion_matrix(xgb_mod_po, xgb_X_po_test, xgb_y_po_test, ax=ax1)
ax1.title.set_text('XGB Hypothyroidism')

plot_confusion_matrix(xgb_mod_per, xgb_X_per_test, xgb_y_per_test, ax=ax2)
ax2.title.set_text('XGB Hyperthyroidism')

plot_confusion_matrix(svm_mod_po, svm_X_po_test, svm_y_po_test, ax=ax3)
ax3.title.set_text('SVM Hypothyroidism')

plot_confusion_matrix(svm_mod_per, svm_X_per_test, svm_y_per_test, ax=ax4)
ax4.title.set_text('SVM Hyperthyroidism')

plot_confusion_matrix(nn_mod_po, nn_X_po_test, nn_y_po_test, ax=ax5)
ax5.title.set_text('MLP Hypothyroidism')

plot_confusion_matrix(nn_mod_per, nn_X_per_test, nn_y_per_test, ax=ax6)
ax6.title.set_text('MLP Hyperthyroidism')
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\classification\\result_data\\S0\\S0_CM.png')
plt.show()


explainerXGB_po = shap.TreeExplainer(xgb_mod_po)
shap_values_XGB_po_test = explainerXGB_po.shap_values(xgb_X_po_test)
df_shap_XGB_po_test = pd.DataFrame(shap_values_XGB_po_test, columns=xgb_X_po_test.columns.values)
test_splt_po_fig = shap.summary_plot(shap_values_XGB_po_test, xgb_X_po_test, show=False)
plt.title("SHIP-0 Hypothyroidism - Feature Importance")
plt.subplots_adjust(left=0.2)
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\classification\\result_data\\S0\\S0_FI_hypo.png')
plt.show()

shap_sum = np.abs(shap_values_XGB_po_test).mean(axis=0)
importance_df_po = pd.DataFrame([xgb_X_po_test.columns.tolist(), shap_sum.tolist()]).T
importance_df_po.columns = ['feature', 'shap_importance']
importance_df_po = importance_df_po.sort_values('shap_importance', ascending=False)

explainerXGB_per = shap.TreeExplainer(xgb_mod_per)
shap_values_XGB_per_test = explainerXGB_per.shap_values(xgb_X_per_test)
df_shap_XGB_per_test = pd.DataFrame(shap_values_XGB_per_test, columns=xgb_X_per_test.columns.values)
test_splt_per_fig = shap.summary_plot(shap_values_XGB_per_test, xgb_X_per_test, show=False)
plt.title("SHIP-0 Hyperthyroidism - Feature Importance")
plt.subplots_adjust(left=0.2)
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\classification\\result_data\\S0\\S0_FI_hyper.png')
plt.show()

shap_sum = np.abs(shap_values_XGB_per_test).mean(axis=0)
importance_df_per = pd.DataFrame([xgb_X_per_test.columns.tolist(), shap_sum.tolist()]).T
importance_df_per.columns = ['feature', 'shap_importance']
importance_df_per = importance_df_per.sort_values('shap_importance', ascending=False)


importance_df_po.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\classification\\result_data\\S0\\S0_imp_hypo.csv")
importance_df_per.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\classification\\result_data\\S0\\S0_imp_hyper.csv")

print(colored("\n...........SHIP0 CLASSIFICATION PLOTTING FINISHED............", "green"))
