import pandas as pd
import statistics
import warnings

# Suppress minor warnings for a clean console output
warnings.filterwarnings("ignore")

# Scikit-learn & Imblearn Imports
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as make_imblearn_pipeline
from imblearn.over_sampling import SMOTE

# The 7 Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

print("==========================================================")
print(" STAGE 1: COMPARING ALL 7 MODELS (GROUP-KFOLD + SMOTE)")
print("==========================================================\n")

# 1. LOAD DATA & ISOLATE THE VAULT
# (We must do this so the models are tested on the exact same 152 instances)
print("Loading data and securing the Holdout Vault...")
X = pd.read_csv("Parkinsons_cleaned.csv")
y = pd.read_csv("Parkinsons_status.csv")
groups = pd.read_csv("Parkinsons_groups.csv").values.ravel() 

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
y_train_flat = y.iloc[train_idx].values.ravel()
groups_train = groups[train_idx] 

print(f"Training Data Secured: {X_train.shape[0]} instances.")
print("-" * 50)

# 2. DEFINE THE 7 MODELS
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# 3. SETUP VALIDATION STRATEGY
cv_strategy = GroupKFold(n_splits=5)
scoring_metrics = ['accuracy', 'f1', 'roc_auc', 'recall']
comparison_results = []

# 4. RUN THE PIPELINE FOR EVERY MODEL
for name, model in classifiers.items():
    print(f"Evaluating {name}...")
    
    # Create the strict leakage-free pipeline for THIS specific model
    pipeline = make_imblearn_pipeline(
        StandardScaler(),
        SMOTE(random_state=42), 
        model
    )
    
    # Run 5-Fold Group Cross Validation on the Training Data ONLY
    cv_results = cross_validate(pipeline, X_train, y_train_flat, 
                                groups=groups_train, 
                                cv=cv_strategy, 
                                scoring=scoring_metrics)
    
    # Save the mean scores
    comparison_results.append({
        "Model": name,
        "Accuracy": statistics.mean(cv_results['test_accuracy']),
        "F1-Score": statistics.mean(cv_results['test_f1']),
        "Recall": statistics.mean(cv_results['test_recall']),
        "ROC-AUC": statistics.mean(cv_results['test_roc_auc'])
    })

# 5. PRINT THE FINAL LEADERBOARD
results_df = pd.DataFrame(comparison_results).sort_values(by="F1-Score", ascending=False)

print("\n==========================================================")
print(" MODEL LEADERBOARD (Sorted by F1-Score)")
print("==========================================================")
print(results_df.to_string(index=False))
print("==========================================================\n")