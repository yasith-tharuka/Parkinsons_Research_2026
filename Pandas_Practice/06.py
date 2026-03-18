import pandas as pd

df = pd.read_csv("test.csv")
#Filtering = keeping the rows that match a condition

#tall_pokemon = df[df["Height"]>=2]
#print(tall_pokemon)

#print("---------")
#legendery_pokemon = df[df["Legendary"] == 1] 
#print(legendery_pokemon)

water_pokemon = df[(df["Type1"] == "Water") |
                   (df["Type2"] == "Water")]

print(water_pokemon)               
