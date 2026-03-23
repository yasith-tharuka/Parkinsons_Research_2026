import pandas as pd
from sklearn.model_selection import train_test_split

#Load Files
X= pd.read_csv("Parkinsons_cleaned.csv")
y=pd.read_csv("Parkinsons_status.csv")

#Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, #20% Test Data(39), 80% train Data(156)
    random_state=42,
    stratify=y
)

print(f"Traning Set: {X_train.shape[0]}")
print(f"Testing set: {X_test.shape[0]}")