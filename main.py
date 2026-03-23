import pandas as pd
import statistics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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

# PART 1:(Data Loading & Split)
X = pd.read_csv("Parkinsons_cleaned.csv")
y = pd.read_csv("Parkinsons_status.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training Set (For 5-Fold CV): {X_train.shape[0]}")
print(f"Testing set (The Vault): {X_test.shape[0]}")
print("----------------------")

# Flatten y_train(get them into one row instead of previuos long column)
y_train_flat = y_train.values.ravel()


# PART 2: THE 5-FOLD CROSS VALIDATION (On X_train only!)
print("Initializing 7 Classifiers in Pipelines")


models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=42)),
    "Decision Tree": make_pipeline(StandardScaler(), DecisionTreeClassifier(class_weight='balanced', random_state=42)),
    "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(class_weight='balanced', random_state=42)),
    "SVM (Linear)": make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced', random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "XGBoost": make_pipeline(StandardScaler(), XGBClassifier(scale_pos_weight=3, random_state=42, eval_metric='logloss')) 
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'f1', 'roc_auc', 'recall'] 

results = []
print("Running 5-Fold Cross-Validation... (Please wait)\n")

for name, pipeline in models.items():
    # Notice we only pass X_train here! The vault (X_test) is safe.
    cv_results = cross_validate(pipeline, X_train, y_train_flat, 
                                cv=cv_strategy, scoring=scoring_metrics)
    
    results.append({
        "Model": name,
        "Train Time (s)": round(statistics.mean(cv_results['fit_time']), 5),
        "Accuracy": round(statistics.mean(cv_results['test_accuracy']), 3),
        "F1-Score": round(statistics.mean(cv_results['test_f1']), 3),
        "ROC-AUC": round(statistics.mean(cv_results['te st_roc_auc']), 3),
        "Sensitivity": round(statistics.mean(cv_results['test_recall']), 3) 
})

results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print(results_df.to_string(index=False))