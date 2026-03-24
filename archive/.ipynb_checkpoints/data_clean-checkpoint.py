import pandas as pd

#Load data
df = pd.read_csv("parkinsons.data")

#----------Check-----
#Check for missing values
print("Missing Values: ", df.isnull().sum().sum())
#(if need to clear)df = df.dropna()

#check for duplicates
print(f"Duplicate Rows: {df.duplicated().sum()}")
#(if need to clear)df = df.drop_duplicates()

#-----------sepration---------
#Features (exept name and status)
features = df.drop(columns=["name","status"])

#Status (Answers)
status = df["status"]

#-----Output----
print(f"Features count : {features.shape[1]}")
print(f"Sample count : {features.shape[0]}")
print(f"-- Status --\n{status.head()}")

#-----Create New file with the cleaned data----
features.to_csv("parkinsons_cleaned.csv",index=False)

