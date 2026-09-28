# pandas 
''''
import pandas as pd
marks = pd.Series([70, 75 ,85, 90])

data ={
    "Name":["Rahul","Priya","Amit"],
    "Marks":[70,80,90]
    }
df = pd.DataFrame (data)

df = df[df["Marks"] >80]

print (df)
'''



import pandas as pd
df= pd.read_csv('customer_spending.csv')
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
res=df[(df["Age"]>60) & (df["Annual_Income"]>100000)]


print(res)
