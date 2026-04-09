import pandas as pd

# Load data
df = pd.read_csv("parkinsons.data")

# ---------- NEW: Extract Patient ID ----------
# The 'name' column contains strings like 'phon_R01_S01_1'
# We split by '_' and grab the 3rd part ('S01') to act as our Patient ID
df['patient_id'] = df['name'].apply(lambda x: x.split('_')[2])

# ----------- Separation ---------
# Features (drop name, status, AND the new patient_id from the math data)
features = df.drop(columns=["name", "status", "patient_id"])

# Status (Answers)
status = df["status"]

# Groups (The Patient IDs we just extracted)
groups = df["patient_id"]

# ----- Create New files ----
features.to_csv("Parkinsons_cleaned.csv", index=False)
status.to_csv("Parkinsons_status.csv", index=False)
groups.to_csv("Parkinsons_groups.csv", index=False) # We will load this in the main script!

print("Data cleaning complete. 3 files generated.")