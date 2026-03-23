import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# The 7 Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# --- 1. PREPARATION ---
# Make sure X and y are loaded from your CSVs first!
# X = pd.read_csv("Parkinsons_cleaned.csv")
# y = pd.read_csv("Parkinsons_status.csv")
y_flat = y.values.ravel() # Flatten for Scikit-Learn

print("\n--- Initializing 7 Classifiers in Pipelines ---")

# --- 2. THE PIPELINES ---
# Notice how StandardScaler() is bundled INSIDE the model. 
# This guarantees it scales perfectly during every single fold without data leakage!
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=42)),
    "Decision Tree": make_pipeline(StandardScaler(), DecisionTreeClassifier(class_weight='balanced', random_state=42)),
    "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(class_weight='balanced', random_state=42)),
    "SVM (Linear)": make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced', random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "XGBoost": make_pipeline(StandardScaler(), XGBClassifier(scale_pos_weight=3, random_state=42, eval_metric='logloss')) 
}

# --- 3. THE 5-FOLD STRATIFIED SETUP ---
# n_splits=5 is your K=5!
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'f1', 'roc_auc', 'recall'] 

results = []
print("Running 5-Fold Cross-Validation... (Please wait)\n")

# --- 4. THE EVALUATION LOOP ---
for name, pipeline in models.items():
    # We pass the WHOLE dataset (X, y_flat) because the cross_validate function 
    # automatically chops it into the 80/20 folds 5 different times!
    cv_results = cross_validate(pipeline, X, y_flat, 
                                cv=cv_strategy, scoring=scoring_metrics)
    
    # Store the average of the 5 folds
    results.append({
        "Model": name,
        "Train Time (s)": round(np.mean(cv_results['fit_time']), 5),
        "Accuracy": round(np.mean(cv_results['test_accuracy']), 3),
        "F1-Score": round(np.mean(cv_results['test_f1']), 3),
        "ROC-AUC": round(np.mean(cv_results['test_roc_auc']), 3),
        "Sensitivity": round(np.mean(cv_results['test_recall']), 3) # Recall is Sensitivity
    })

# --- 5. DISPLAY RESULTS ---
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print(results_df.to_string(index=False))