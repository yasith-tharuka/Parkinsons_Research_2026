import pandas as pd

df = pd.read_csv("test.csv", index_col="Name")


#Selection by column
#print(df["Name"].to_string())
#print(df[["Name","Height","Weight"]].to_string())

#Selection by Row
#print(df.loc["Pikachu"])
#print(df.loc["Charizard",["Height","Weight"]])
#print(df.loc["Charizard":"Blastoise"])
#print(df.iloc[0:11:2,])
#only first 03 columns
#print(df.iloc[0:11:2, 0:3])

Pokemon = input("Enter a Pokemon name:")

try:
    print(df.loc[Pokemon])
except KeyError:
    print(f"{Pokemon} not found")
