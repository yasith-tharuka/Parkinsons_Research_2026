from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- CRITICAL FIX FOR PANDAS ---
# Because you loaded 'y' from a CSV, it is technically a 2D table (156 rows, 1 column).
# Scikit-Learn models expect a 1D list (156,). .values.ravel() flattens it so the AI doesn't crash.
y_train_flat = y_train.values.ravel()
y_test_flat = y_test.values.ravel()

print("\n--- Model Training & Results ---")

# ==========================================
# MODEL 1: Logistic Regression (The Baseline)
# ==========================================
# We use class_weight='balanced' to handle your 75/25 imbalance
lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train_flat)             # 1. Learn from the training data
lr_predictions = lr_model.predict(X_test_scaled)       # 2. Take the "Final Exam"

lr_acc = accuracy_score(y_test_flat, lr_predictions)
lr_f1 = f1_score(y_test_flat, lr_predictions)
print(f"Logistic Regression -> Accuracy: {lr_acc:.2f} | F1-Score: {lr_f1:.2f}")

# ==========================================
# MODEL 2: Decision Tree (The Explainable Model)
# ==========================================
dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_model.fit(X_train_scaled, y_train_flat)
dt_predictions = dt_model.predict(X_test_scaled)

dt_acc = accuracy_score(y_test_flat, dt_predictions)
dt_f1 = f1_score(y_test_flat, dt_predictions)
print(f"Decision Tree       -> Accuracy: {dt_acc:.2f} | F1-Score: {dt_f1:.2f}")

# ==========================================
# MODEL 3: Random Forest (The Heavyweight Benchmark)
# ==========================================
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train_flat)
rf_predictions = rf_model.predict(X_test_scaled)

rf_acc = accuracy_score(y_test_flat, rf_predictions)
rf_f1 = f1_score(y_test_flat, rf_predictions)
print(f"Random Forest       -> Accuracy: {rf_acc:.2f} | F1-Score: {rf_f1:.2f}")