import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as make_imblearn_pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc

print("==========================================================")
print(" FINAL CLINICAL EVALUATION (OPENING THE VAULT)")
print("==========================================================\n")

# 1. LOAD DATA & ISOLATE THE VAULT 
print("Loading data and securing the Holdout Vault...")
X = pd.read_csv("Parkinsons_cleaned.csv")
y = pd.read_csv("Parkinsons_status.csv")
groups = pd.read_csv("Parkinsons_groups.csv").values.ravel() 

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_flat, y_test_flat = y.iloc[train_idx].values.ravel(), y.iloc[test_idx].values.ravel()

# 2. INITIALIZE THE LEAKAGE-FREE CHAMPION PIPELINE (XGBoost ONLY)
champion_pipeline = make_imblearn_pipeline(
    StandardScaler(),
    SMOTE(random_state=42), 
    XGBClassifier(random_state=42, eval_metric='logloss')
)

# 3. TRAIN ON THE 152 INSTANCES
print("Training Champion Pipeline on the 152 training instances...")
champion_pipeline.fit(X_train, y_train_flat)

# 4. TEST ON THE 43 UNSEEN INSTANCES (THE VAULT)
print("Testing on 43 unseen patient instances...\n")
final_preds = champion_pipeline.predict(X_test)
y_probs = champion_pipeline.predict_proba(X_test)[:, 1]

# Print final metrics
print("--- FINAL METRICS ---")
print(f"Final Accuracy: {accuracy_score(y_test_flat, final_preds):.3f}")
print(f"Final F1-Score: {f1_score(y_test_flat, final_preds):.3f}\n")
print(classification_report(y_test_flat, final_preds, target_names=["Healthy (0)", "Parkinson's (1)"]))

# 5. GENERATE AND SAVE GRAPHS
print("\n--- GENERATING GRAPHS ---")
# --- Figure 1: Confusion Matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_flat, final_preds), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', "Parkinson's"], yticklabels=['Healthy', "Parkinson's"])
plt.title('Final Performance: XGBoost (Group-Aware)')
plt.ylabel('True Diagnosis')
plt.xlabel('Classifier Prediction')
plt.tight_layout()
plt.savefig("Fig1_Confusion_Matrix.png", dpi=300)
print("-> Saved: Fig1_Confusion_Matrix.png")

# --- Figure 2: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test_flat, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Holdout Vault)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Fig2_ROC_Curve.png", dpi=300)
print("-> Saved: Fig2_ROC_Curve.png")

# 6. EXPLAINABLE AI (XAI) BIOMARKER EXTRACTION
print("\n--- EXTRACTING EXPLAINABLE AI BIOMARKERS ---")
xgb_model = champion_pipeline.named_steps['xgbclassifier']
importances = xgb_model.feature_importances_

importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# --- Figure 3: Feature Importance Bar Chart ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis', hue='Feature', legend=False)
plt.title("Top 10 Biomarkers (XGBoost + SMOTE)")
plt.xlabel("Relative Importance")
plt.ylabel("Acoustic Feature")
plt.tight_layout()
plt.savefig("Fig3_Feature_Importance.png", dpi=300)
print("-> Saved: Fig3_Feature_Importance.png")

print("\n--- TOP 5 DRIVERS OF DIAGNOSIS ---")
print(importance_df.head(5).to_string(index=False))
print("\n*** SCRIPT COMPLETE ***")