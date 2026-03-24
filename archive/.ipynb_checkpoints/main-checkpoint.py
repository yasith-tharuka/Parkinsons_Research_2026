from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Define the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# 2. Loop through and train each one
print(f"{'Model':<20} | {'Accuracy':<10} | {'F1-Score':<10}")
print("-" * 45)

for name, model in models.items():
    # Training (This happens instantly)
    model.fit(X_train_scaled, y_train)
    
    # Testing
    y_pred = model.predict(X_test_scaled)
    
    # Calculating Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{name:<20} | {acc:>10.2%} | {f1:>10.2%}")