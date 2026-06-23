import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as make_imblearn_pipeline
from imblearn.over_sampling import SMOTE

# The 7 Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# --- 1. PREPARATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

print("Loading High-Quality V2 data...")
X = pd.read_csv(DATA_DIR / "Parkinsons_v2_features.csv")
y = pd.read_csv(DATA_DIR / "Parkinsons_v2_status.csv")
groups = pd.read_csv(DATA_DIR / "Parkinsons_v2_groups.csv").values.ravel()
y_flat = y.values.ravel()

print("\n--- Initializing 7 Classifiers in Pipelines with SMOTE ---")

# --- 2. THE PIPELINES ---
models = {
    "Logistic Regression": make_imblearn_pipeline(StandardScaler(), SMOTE(random_state=42), LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)),
    "Decision Tree": make_imblearn_pipeline(StandardScaler(), SMOTE(random_state=42), DecisionTreeClassifier(class_weight='balanced', random_state=42)),
    "Random Forest": make_imblearn_pipeline(StandardScaler(), SMOTE(random_state=42), RandomForestClassifier(class_weight='balanced', random_state=42)),
    "SVM (Linear)": make_imblearn_pipeline(StandardScaler(), SMOTE(random_state=42), SVC(kernel='linear', class_weight='balanced', random_state=42)),
    "KNN": make_imblearn_pipeline(StandardScaler(), SMOTE(random_state=42), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": make_imblearn_pipeline(StandardScaler(), SMOTE(random_state=42), GaussianNB()),
    "XGBoost": make_imblearn_pipeline(StandardScaler(), SMOTE(random_state=42), XGBClassifier(scale_pos_weight=3, random_state=42, eval_metric='logloss')) 
}

# --- 3. THE 5-FOLD STRATIFIED SETUP ---
cv_strategy = GroupKFold(n_splits=5)
scoring_metrics = ['accuracy', 'f1', 'roc_auc', 'recall', 'balanced_accuracy'] 

results = []
print(f"Running 5-Fold Group Cross-Validation on {X.shape[0]} instances from {len(np.unique(groups))} patients... (Please wait)\n")

# --- 4. THE EVALUATION LOOP ---
for name, pipeline in models.items():
    print(f"Training {name}...")
    cv_results = cross_validate(pipeline, X, y_flat, groups=groups,
                                cv=cv_strategy, scoring=scoring_metrics, n_jobs=-1)
    
    # Store the average of the 5 folds
    results.append({
        "Model": name,
        "Balanced Accuracy": round(np.mean(cv_results['test_balanced_accuracy']), 3),
        "F1-Score": round(np.mean(cv_results['test_f1']), 3),
        "ROC-AUC": round(np.mean(cv_results['test_roc_auc']), 3),
        "Sensitivity": round(np.mean(cv_results['test_recall']), 3)
    })

# --- 5. DISPLAY RESULTS ---
print("\n=======================================================================")
print("          MODEL COMPARISON ON HIGH-QUALITY V2 DATASET")
print("=======================================================================")
results_df = pd.DataFrame(results).sort_values(by="Balanced Accuracy", ascending=False)
print(results_df.to_string(index=False))
print("=======================================================================")
