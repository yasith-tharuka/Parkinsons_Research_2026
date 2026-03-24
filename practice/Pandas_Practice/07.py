import pandas as pd

#Aggeregate Functions = Reduces a set of values into a single summery value
#                       Used to summerize and analyze data
#                       Ofted used with the groupby() function

df = pd.read_csv("test.csv")


#---Whole Dataframe----
#print(df.mean(numeric_only=True))
#print(df.sum(numeric_only=True))
#print(df.min(numeric_only=True))
#print(df.max(numeric_only=True))
#print(df.count()) #Doesnt count NULL values

#---Single Column---
#print(df["Height"].mean())
#print(df["Height"].sum())
#print(df["Height"].min())
#print(df["Height"].max())
#print(df["Height"].count()) #Doesnt count NULL values

#Average Height group by Type1
group = df.groupby("Type1")
print(group["Height"].mean())