import pandas as pd

# 1. Load the specific file we identified
# Ensure the file 'parkinsons.data' is in the same folder as this script
df = pd.read_csv('parkinsons.data')

# 2. Peek at the data
# This will show you the column names like 'MDVP:Fo(Hz)', 'Jitter:Local(%)', and 'status'
print("Dataset successfully loaded!")
print(f"Total entries: {len(df)}")
print("\nFirst 3 rows of data:")
print(df.head(3))

# 3. Verify the target
# 'status' is our answer key: 1 for Parkinson's, 0 for Healthy
print("\nCheck if we have both classes (0 and 1):")
print(df['status'].value_counts())