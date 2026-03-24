from sklearn.preprocessing import StandardScaler

# 1. Initialize the Scaler
scaler = StandardScaler()

# 2. 'Fit' and 'Transform' the Training Data
# The AI learns the scale of the 156 training rows and standardizes them.
X_train_scaled = scaler.fit_transform(X_train)

# 3. ONLY 'Transform' the Testing Data
# We apply the learned scale to the 39 testing rows (No peeking allowed!)
X_test_scaled = scaler.transform(X_test)

print("--- Preprocessing Complete ---")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print("Data is scaled and ready for the AI models!")