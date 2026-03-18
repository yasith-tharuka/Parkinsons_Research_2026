import pandas as pd

#use dictionaries
calories = {"Day 1":1750,
            "Day 2":2100,
            "Day 3":1700}

series = pd.Series(calories)

print(series)
 