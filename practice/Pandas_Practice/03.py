import pandas as pd
data = {"Name":["Spingebob","Patric","Squidward"],
        "Age":[30,35,50]}

df = pd.DataFrame(data,index=["Emp 1","Emp 2","Emp 3"])
#Add a new column
df["Job"] = ["Cook","N/A","Cashier"]
#add a new row
new_row = pd.DataFrame([{"Name":"Sandy","Age":28,"Job":"Engineer"},
                        {"Name":"Eugene","Age":16,"Job":"Manager"}],
                       index=["Emp 4","Emp 5"])
df = pd.concat([df,new_row])

#print(df.loc["Emp 2"])
#print(df.iloc[0])

print(df)
