import matplotlib.pyplot as plt
import shap

from analysis.regression.models.regression_models import *

print(colored("\n...........REGRESSION PLOTTING STARTED............", "green"))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3,
                                                                                                                sharey=True)
ax1.scatter(y0_train, S0_xgb_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax1.scatter(y0_test, S0_xgb_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax1.plot([y0.min(), y0.max()], [y0.min(), y0.max()], 'k--', lw=2)
ax1.set_xlabel('Measured FT4 (pmol/l)')
ax1.set_ylabel('Predicted FT4 (pmol/l)')
ax1.set_title("XGB")
ax1.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S0_xgb_R2_train, 3)), c="#0080ff")
ax1.text(0.07, 0.9, "Train MAE = {}".format(round(S0_xgb_MAE_train, 3)), c="#0080ff")
ax1.text(0.07, 0.85, "Train RMSE = {}".format(round(S0_xgb_RMSE_train, 3)), c="#0080ff")
ax1.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S0_xgb_R2_test, 3)), c="#F1455F")
ax1.text(0.07, 0.7, "Test MAE = {}".format(round(S0_xgb_MAE_test, 3)), c="#F1455F")
ax1.text(0.07, 0.65, "Test RMSE = {}".format(round(S0_xgb_RMSE_test, 3)), c="#F1455F")
ax1.text(0.07, 0.58, "Target Std = {}".format(round(S0_target_std, 3)))
ax1.set(aspect='equal')
fig.suptitle('REGRESSION 3', fontsize=16)

ax2.scatter(y0_train, S0_svr_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax2.scatter(y0_test, S0_svr_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax2.plot([y0.min(), y0.max()], [y0.min(), y0.max()], 'k--', lw=2)
ax2.set_xlabel('Measured FT4 (pmol/l)')
ax2.set_title("SVM")
ax2.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S0_svr_R2_train, 3)), c="#0080ff")
ax2.text(0.07, 0.9, "Train MAE = {}".format(round(S0_svr_MAE_train, 3)), c="#0080ff")
ax2.text(0.07, 0.85, "Train RMSE = {}".format(round(S0_svr_RMSE_train, 3)), c="#0080ff")
ax2.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S0_svr_R2_test, 3)), c="#F1455F")
ax2.text(0.07, 0.7, "Test MAE = {}".format(round(S0_svr_MAE_test, 3)), c="#F1455F")
ax2.text(0.07, 0.65, "Test RMSE = {}".format(round(S0_svr_RMSE_test, 3)), c="#F1455F")
ax2.text(0.07, 0.58, "Target Std = {}".format(round(S0_target_std, 3)))
ax2.set(aspect='equal')

ax3.scatter(y0_train, S0_nn_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax3.scatter(y0_test, S0_nn_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax3.plot([y0.min(), y0.max()], [y0.min(), y0.max()], 'k--', lw=2)
ax3.set_xlabel('Measured FT4 (pmol/l)')
ax3.set_title("MLP")
ax3.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S0_nn_R2_train, 3)), c="#0080ff")
ax3.text(0.07, 0.9, "Train MAE = {}".format(round(S0_nn_MAE_train, 3)), c="#0080ff")
ax3.text(0.07, 0.85, "Train RMSE = {}".format(round(S0_nn_RMSE_train, 3)), c="#0080ff")
ax3.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S0_nn_R2_test, 3)), c="#F1455F")
ax3.text(0.07, 0.7, "Test MAE = {}".format(round(S0_nn_MAE_test, 3)), c="#F1455F")
ax3.text(0.07, 0.65, "Test RMSE = {}".format(round(S0_nn_RMSE_test, 3)), c="#F1455F")
ax3.text(0.07, 0.58, "Target Std = {}".format(round(S0_target_std, 3)))
ax3.set(aspect='equal')

ax4.scatter(y1_train, S1_xgb_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax4.scatter(y1_test, S1_xgb_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax4.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=2)
ax4.set_xlabel('Measured FT4 (pmol/l)')
ax4.set_ylabel('Predicted FT4 (pmol/l)')
ax4.set_title("XGB")
ax4.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S1_xgb_R2_train, 3)), c="#0080ff")
ax4.text(0.07, 0.9, "Train MAE = {}".format(round(S1_xgb_MAE_train, 3)), c="#0080ff")
ax4.text(0.07, 0.85, "Train RMSE = {}".format(round(S1_xgb_RMSE_train, 3)), c="#0080ff")
ax4.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S1_xgb_R2_test, 3)), c="#F1455F")
ax4.text(0.07, 0.7, "Test MAE = {}".format(round(S1_xgb_MAE_test, 3)), c="#F1455F")
ax4.text(0.07, 0.65, "Test RMSE = {}".format(round(S1_xgb_RMSE_test, 3)), c="#F1455F")
ax4.text(0.07, 0.58, "Target Std = {}".format(round(S1_target_std, 3)))
ax4.set(aspect='equal')

ax5.scatter(y1_train, S1_svr_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax5.scatter(y1_test, S1_svr_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax5.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=2)
ax5.set_xlabel('Measured FT4 (pmol/l)')
ax5.set_title("SVM")
ax5.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S1_svr_R2_train, 3)), c="#0080ff")
ax5.text(0.07, 0.9, "Train MAE = {}".format(round(S1_svr_MAE_train, 3)), c="#0080ff")
ax5.text(0.07, 0.85, "Train RMSE = {}".format(round(S1_svr_RMSE_train, 3)), c="#0080ff")
ax5.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S1_svr_R2_test, 3)), c="#F1455F")
ax5.text(0.07, 0.7, "Test MAE = {}".format(round(S1_svr_MAE_test, 3)), c="#F1455F")
ax5.text(0.07, 0.65, "Test RMSE = {}".format(round(S1_svr_RMSE_test, 3)), c="#F1455F")
ax5.text(0.07, 0.58, "Target Std = {}".format(round(S1_target_std, 3)))
ax5.set(aspect='equal')

ax6.scatter(y1_train, S1_nn_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax6.scatter(y1_test, S1_nn_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax6.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'k--', lw=2)
ax6.set_xlabel('Measured FT4 (pmol/l)')
ax6.set_title("MLP")
ax6.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S1_nn_R2_train, 3)), c="#0080ff")
ax6.text(0.07, 0.9, "Train MAE = {}".format(round(S1_nn_MAE_train, 3)), c="#0080ff")
ax6.text(0.07, 0.85, "Train RMSE = {}".format(round(S1_nn_RMSE_train, 3)), c="#0080ff")
ax6.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S1_nn_R2_test, 3)), c="#F1455F")
ax6.text(0.07, 0.7, "Test MAE = {}".format(round(S1_nn_MAE_test, 3)), c="#F1455F")
ax6.text(0.07, 0.65, "Test RMSE = {}".format(round(S1_nn_RMSE_test, 3)), c="#F1455F")
ax6.text(0.07, 0.58, "Target Std = {}".format(round(S1_target_std, 3)))
ax6.set(aspect='equal')

ax7.scatter(y2_train, S2_xgb_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax7.scatter(y2_test, S2_xgb_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax7.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'k--', lw=2)
ax7.set_xlabel('Measured FT4 (pmol/l)')
ax7.set_ylabel('Predicted FT4 (pmol/l)')
ax7.set_title("XGB")
ax7.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S2_xgb_R2_train, 3)), c="#0080ff")
ax7.text(0.07, 0.9, "Train MAE = {}".format(round(S2_xgb_MAE_train, 3)), c="#0080ff")
ax7.text(0.07, 0.85, "Train RMSE = {}".format(round(S2_xgb_RMSE_train, 3)), c="#0080ff")
ax7.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S2_xgb_R2_test, 3)), c="#F1455F")
ax7.text(0.07, 0.7, "Test MAE = {}".format(round(S2_xgb_MAE_test, 3)), c="#F1455F")
ax7.text(0.07, 0.65, "Test RMSE = {}".format(round(S2_xgb_RMSE_test, 3)), c="#F1455F")
ax7.text(0.07, 0.58, "Target Std = {}".format(round(S2_target_std, 3)))
ax7.set(aspect='equal')

ax8.scatter(y2_train, S2_svr_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax8.scatter(y2_test, S2_svr_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax8.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'k--', lw=2)
ax8.set_xlabel('Measured FT4 (pmol/l)')
ax8.set_title("SVM")
ax8.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S2_svr_R2_train, 3)), c="#0080ff")
ax8.text(0.07, 0.9, "Train MAE = {}".format(round(S2_svr_MAE_train, 3)), c="#0080ff")
ax8.text(0.07, 0.85, "Train RMSE = {}".format(round(S2_svr_RMSE_train, 3)), c="#0080ff")
ax8.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S2_svr_R2_test, 3)), c="#F1455F")
ax8.text(0.07, 0.7, "Test MAE = {}".format(round(S2_svr_MAE_test, 3)), c="#F1455F")
ax8.text(0.07, 0.65, "Test RMSE = {}".format(round(S2_svr_RMSE_test, 3)), c="#F1455F")
ax8.text(0.07, 0.58, "Target Std = {}".format(round(S2_target_std, 3)))
ax8.set(aspect='equal')

ax9.scatter(y2_train, S2_nn_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax9.scatter(y2_test, S2_nn_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax9.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'k--', lw=2)
ax9.set_xlabel('Measured FT4 (pmol/l)')
ax9.set_title("MLP")
ax9.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S2_nn_R2_train, 3)), c="#0080ff")
ax9.text(0.07, 0.9, "Train MAE = {}".format(round(S2_nn_MAE_train, 3)), c="#0080ff")
ax9.text(0.07, 0.85, "Train RMSE = {}".format(round(S2_nn_RMSE_train, 3)), c="#0080ff")
ax9.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S2_nn_R2_test, 3)), c="#F1455F")
ax9.text(0.07, 0.7, "Test MAE = {}".format(round(S2_nn_MAE_test, 3)), c="#F1455F")
ax9.text(0.07, 0.65, "Test RMSE = {}".format(round(S2_nn_RMSE_test, 3)), c="#F1455F")
ax9.text(0.07, 0.58, "Target Std = {}".format(round(S2_target_std, 3)))
ax9.set(aspect='equal')

ax10.scatter(y3_train, S3_xgb_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax10.scatter(y3_test, S3_xgb_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax10.plot([y3.min(), y3.max()], [y3.min(), y3.max()], 'k--', lw=2)
ax10.set_xlabel('Measured FT4 (pmol/l)')
ax10.set_ylabel('Predicted FT4 (pmol/l)')
ax10.set_title("XGB")
ax10.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S3_xgb_R2_train, 3)), c="#0080ff")
ax10.text(0.07, 0.9, "Train MAE = {}".format(round(S3_xgb_MAE_train, 3)), c="#0080ff")
ax10.text(0.07, 0.85, "Train RMSE = {}".format(round(S3_xgb_RMSE_train, 3)), c="#0080ff")
ax10.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S3_xgb_R2_test, 3)), c="#F1455F")
ax10.text(0.07, 0.7, "Test MAE = {}".format(round(S3_xgb_MAE_test, 3)), c="#F1455F")
ax10.text(0.07, 0.65, "Test RMSE = {}".format(round(S3_xgb_RMSE_test, 3)), c="#F1455F")
ax10.text(0.07, 0.58, "Target Std = {}".format(round(S3_target_std, 3)))
ax10.set(aspect='equal')

ax11.scatter(y3_train, S3_svr_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax11.scatter(y3_test, S3_svr_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax11.plot([y3.min(), y3.max()], [y3.min(), y3.max()], 'k--', lw=2)
ax11.set_xlabel('Measured FT4 (pmol/l)')
ax11.set_title("SVM")
ax11.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S3_svr_R2_train, 3)), c="#0080ff")
ax11.text(0.07, 0.9, "Train MAE = {}".format(round(S3_svr_MAE_train, 3)), c="#0080ff")
ax11.text(0.07, 0.85, "Train RMSE = {}".format(round(S3_svr_RMSE_train, 3)), c="#0080ff")
ax11.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S3_svr_R2_test, 3)), c="#F1455F")
ax11.text(0.07, 0.7, "Test MAE = {}".format(round(S3_svr_MAE_test, 3)), c="#F1455F")
ax11.text(0.07, 0.65, "Test RMSE = {}".format(round(S3_svr_RMSE_test, 3)), c="#F1455F")
ax11.text(0.07, 0.58, "Target Std = {}".format(round(S3_target_std, 3)))
ax11.set(aspect='equal')

ax12.scatter(y3_train, S3_nn_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax12.scatter(y3_test, S3_nn_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax12.plot([y3.min(), y3.max()], [y3.min(), y3.max()], 'k--', lw=2)
ax12.set_xlabel('Measured FT4 (pmol/l)')
ax12.set_title("MLP")
ax12.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(S3_nn_R2_train, 3)), c="#0080ff")
ax12.text(0.07, 0.9, "Train MAE = {}".format(round(S3_nn_MAE_train, 3)), c="#0080ff")
ax12.text(0.07, 0.85, "Train RMSE = {}".format(round(S3_nn_RMSE_train, 3)), c="#0080ff")
ax12.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(S3_nn_R2_test, 3)), c="#F1455F")
ax12.text(0.07, 0.7, "Test MAE = {}".format(round(S3_nn_MAE_test, 3)), c="#F1455F")
ax12.text(0.07, 0.65, "Test RMSE = {}".format(round(S3_nn_RMSE_test, 3)), c="#F1455F")
ax12.text(0.07, 0.58, "Target Std = {}".format(round(S3_target_std, 3)))
ax12.set(aspect='equal')

ax13.scatter(yT_train, T0_xgb_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax13.scatter(yT_test, T0_xgb_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax13.plot([yT.min(), yT.max()], [yT.min(), yT.max()], 'k--', lw=2)
ax13.set_xlabel('Measured FT4 (pmol/l)')
ax13.set_ylabel('Predicted FT4 (pmol/l)')
ax13.set_title("XGB")
ax13.text(0.07, 1, "Train R\u00b2 = {}".format(round(T0_xgb_R2_train, 3)), c="#0080ff")
ax13.text(0.07, 0.95, "Train MAE = {}".format(round(T0_xgb_MAE_train, 3)), c="#0080ff")
ax13.text(0.07, 0.9, "Train RMSE = {}".format(round(T0_xgb_RMSE_train, 3)), c="#0080ff")
ax13.text(0.07, 0.8, "Test R\u00b2 = {}".format(round(T0_xgb_R2_test, 3)), c="#F1455F")
ax13.text(0.07, 0.75, "Test MAE = {}".format(round(T0_xgb_MAE_test, 3)), c="#F1455F")
ax13.text(0.07, 0.70, "Test RMSE = {}".format(round(T0_xgb_RMSE_test, 3)), c="#F1455F")
ax13.text(0.07, 0.63, "Target Std = {}".format(round(T0_target_std, 3)))
ax13.set(aspect='equal')

ax14.scatter(yT_train, T0_svr_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax14.scatter(yT_test, T0_svr_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax14.plot([yT.min(), yT.max()], [yT.min(), yT.max()], 'k--', lw=2)
ax14.set_xlabel('Measured FT4 (pmol/l)')
ax14.set_title("SVM")
ax14.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(T0_svr_R2_train, 3)), c="#0080ff")
ax14.text(0.07, 0.9, "Train MAE = {}".format(round(T0_svr_MAE_train, 3)), c="#0080ff")
ax14.text(0.07, 0.85, "Train RMSE = {}".format(round(T0_svr_RMSE_train, 3)), c="#0080ff")
ax14.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(T0_svr_R2_test, 3)), c="#F1455F")
ax14.text(0.07, 0.7, "Test MAE = {}".format(round(T0_svr_MAE_test, 3)), c="#F1455F")
ax14.text(0.07, 0.65, "Test RMSE = {}".format(round(T0_svr_RMSE_test, 3)), c="#F1455F")
ax14.text(0.07, 0.58, "Target Std = {}".format(round(T0_target_std, 3)))
ax14.set(aspect='equal')

ax15.scatter(yT_train, T0_nn_train_preds, s=20, edgecolors="#0080ff", label="train", facecolors='none')
ax15.scatter(yT_test, T0_nn_test_preds, s=20, edgecolors="#F1455F", label="test", facecolors='none')
ax15.plot([yT.min(), yT.max()], [yT.min(), yT.max()], 'k--', lw=2)
ax15.set_xlabel('Measured FT4 (pmol/l)')
ax15.set_ylabel('Predicted FT4 (pmol/l)')
ax15.set_title("MLP")
ax15.text(0.07, 0.95, "Train R\u00b2 = {}".format(round(T0_nn_R2_train, 3)), c="#0080ff")
ax15.text(0.07, 0.9, "Train MAE = {}".format(round(T0_nn_MAE_train, 3)), c="#0080ff")
ax15.text(0.07, 0.85, "Train RMSE = {}".format(round(T0_nn_RMSE_train, 3)), c="#0080ff")
ax15.text(0.07, 0.75, "Test R\u00b2 = {}".format(round(T0_nn_R2_test, 3)), c="#F1455F")
ax15.text(0.07, 0.7, "Test MAE = {}".format(round(T0_nn_MAE_test, 3)), c="#F1455F")
ax15.text(0.07, 0.65, "Test RMSE = {}".format(round(T0_nn_RMSE_test, 3)), c="#F1455F")
ax15.text(0.07, 0.58, "Target Std = {}".format(round(T0_target_std, 3)))
ax15.set(aspect='equal')

fig.set_size_inches(14, 25)
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\reg_parsity_plots.png')
plt.show()

print(colored("\n...........REGRESSION PLOTTING FINISHED............", "green"))


print(colored("\n...........SHAP PLOTTING STARTED............", "green"))

# SHIP-0 Summary plot

S0_explainerXGB = shap.TreeExplainer(S0_xgb_mod)
S0_shap_values_XGB_test = S0_explainerXGB.shap_values(X0_test)
S0_df_shap_XGB_test = pd.DataFrame(S0_shap_values_XGB_test, columns=X0_test.columns.values)
S0_test_splt_fig = shap.summary_plot(S0_shap_values_XGB_test, X0_test, show=False)
plt.title("SHIP-0 Regression - Feature Importance")
plt.subplots_adjust(left=0.2)
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\S0_feature_imp.png')
plt.show()

# SHIP-0 all feature importance dataframe

S0_shap_sum = np.abs(S0_shap_values_XGB_test).mean(axis=0)
S0_importance_df = pd.DataFrame([X0_test.columns.tolist(), S0_shap_sum.tolist()]).T
S0_importance_df.columns = ['feature', 'shap_importance']
S0_importance_df = S0_importance_df.sort_values('shap_importance', ascending=False)



# SHIP-1 Summary plot

S1_explainerXGB = shap.TreeExplainer(S1_xgb_mod)
S1_shap_values_XGB_test = S1_explainerXGB.shap_values(X1_test)
S1_df_shap_XGB_test = pd.DataFrame(S1_shap_values_XGB_test, columns=X1_test.columns.values)
S1_test_splt_fig = shap.summary_plot(S1_shap_values_XGB_test, X1_test, show=False)
plt.title("SHIP-1 Regression - Feature Importance")
plt.subplots_adjust(left=0.2)
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\S1_feature_imp.png')
plt.show()

# SHIP-1 all feature importance dataframe

S1_shap_sum = np.abs(S1_shap_values_XGB_test).mean(axis=0)
S1_importance_df = pd.DataFrame([X1_test.columns.tolist(), S1_shap_sum.tolist()]).T
S1_importance_df.columns = ['feature', 'shap_importance']
S1_importance_df = S1_importance_df.sort_values('shap_importance', ascending=False)

# SHIP-2 Summary plot

S2_explainerXGB = shap.TreeExplainer(S2_xgb_mod)
S2_shap_values_XGB_test = S2_explainerXGB.shap_values(X2_test)
S2_df_shap_XGB_test = pd.DataFrame(S2_shap_values_XGB_test, columns=X2_test.columns.values)
S2_test_splt_fig = shap.summary_plot(S2_shap_values_XGB_test, X2_test, show=False)
plt.title("SHIP-2 Regression - Feature Importance")
plt.subplots_adjust(left=0.2)
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\S2_feature_imp.png')
plt.show()

# SHIP-2 all feature importance dataframe

S2_shap_sum = np.abs(S2_shap_values_XGB_test).mean(axis=0)
S2_importance_df = pd.DataFrame([X2_test.columns.tolist(), S2_shap_sum.tolist()]).T
S2_importance_df.columns = ['feature', 'shap_importance']
S2_importance_df = S2_importance_df.sort_values('shap_importance', ascending=False)

# SHIP-3 Summary plot

S3_explainerXGB = shap.TreeExplainer(S3_xgb_mod)
S3_shap_values_XGB_test = S3_explainerXGB.shap_values(X3_test)
S3_df_shap_XGB_test = pd.DataFrame(S3_shap_values_XGB_test, columns=X3_test.columns.values)
S3_test_splt_fig = shap.summary_plot(S3_shap_values_XGB_test, X3_test, show=False)
plt.title("SHIP-3 Regression - Feature Importance")
plt.subplots_adjust(left=0.2)
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\S3_feature_imp.png')
plt.show()

# SHIP-3 all feature importance dataframe

S3_shap_sum = np.abs(S3_shap_values_XGB_test).mean(axis=0)
S3_importance_df = pd.DataFrame([X3_test.columns.tolist(), S3_shap_sum.tolist()]).T
S3_importance_df.columns = ['feature', 'shap_importance']
S3_importance_df = S3_importance_df.sort_values('shap_importance', ascending=False)

# TREND-0 Summary plot

T0_explainerXGB = shap.TreeExplainer(T0_xgb_mod)
T0_shap_values_XGB_test = T0_explainerXGB.shap_values(XT_test)
T0_df_shap_XGB_test = pd.DataFrame(T0_shap_values_XGB_test, columns=XT_test.columns.values)
T0_test_splt_fig = shap.summary_plot(T0_shap_values_XGB_test, XT_test, show=False)
plt.title("TREND-0 Regression - Feature Importance")
plt.subplots_adjust(left=0.2)
plt.savefig('D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\T0_feature_imp.png')
plt.show()

# TREND-0 all feature importance dataframe

T0_shap_sum = np.abs(T0_shap_values_XGB_test).mean(axis=0)
T0_importance_df = pd.DataFrame([XT_test.columns.tolist(), T0_shap_sum.tolist()]).T
T0_importance_df.columns = ['feature', 'shap_importance']
T0_importance_df = T0_importance_df.sort_values('shap_importance', ascending=False)

S0_importance_df.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\S0_importance.csv")
S1_importance_df.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\S1_importance.csv")
S2_importance_df.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\S2_importance.csv")
S3_importance_df.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\S3_importance.csv")
T0_importance_df.to_csv("D:\\Files\\Work\\Pycharm Projects\\Dissertation\\analysis\\regression\\result_data\\T0_importance.csv")


print(colored("\n...........SHAP PLOTTING FINISHED............", "green"))