import pandas as pd

# Data Cleaning = The process od fixing/remocing:
#                  incomplete, incorrect, or irrelevant data.
#                   ~75% of work done with Pandas is data cleaning


df = pd.read_csv("test.csv")

#1.Drop Irrelevant Columns
#df = df.drop(columns=["Legendary", "No"])

#2.Handle Missing Data
#df = df.dropna(subset=["Type2"])#drop Null values from Type2
#df = df.fillna({"Type2":"None"}) #Fill Null Values from Type2 with a new value

#3.Fix Inxonsistent values
#df["Type1"] = df["Type1"].replace({"Grass":"GRASS",
#                                   "Fire":"FIRE",
#                                   "water":"WATER"})

#4.Standardize text
#df["Name"] = df["Name"].str.lower()

#5.Fix Data Types
#df["Legendary"] = df["Legendary"].astype(bool)

#6. Remove Duplicate values
#df = df.drop_duplicates()

print(df)
