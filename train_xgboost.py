import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

import shap
import seaborn as sns

# Load data
train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

# Define features and labels
features = ["pt1", "eta1", "phi1", "pt2", "eta2", "phi2"]
X_train = train[features]
y_train = train["classID"]
X_test = test[features]
y_test = test["classID"]

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", gamma=3) #default learning_rate=0.3, can also define number of trees by 'n_estimators'(default is 100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
roc_auc_score_value = roc_auc_score(y_test, y_proba)
print("ROC AUC:", roc_auc_score_value)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


booster = model.get_booster()
for i in range(2):  # Show first 2 trees (i.e. tree_idx=0 and 1)
    print(booster.get_dump(with_stats=True)[i])


#the next 2 lines are meant to obtain a visualization of decision trees, but fail due to issue with "pip install graphviz")
#xgb.plot_tree(booster, tree_idx=0, rankdir='LR')  # tree_idx=0 is first tree
#plt.show()


# Plot feature importance
xgb.plot_importance(model)
plt.tight_layout()
plt.show()


# Visualize first tree
#xgb.plot_tree(model, num_trees=0)
#plt.show()



# Plot correlation matrix heatmap
plt.figure(figsize=(8, 6))
corr_matrix = train[features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# --- Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
#plt.savefig("confusion_matrix_heatmap.png")
plt.show()


#SHAP
# SHAP values
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, show=True)

# ROC curve (per class — binary case)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
