import pandas as pd
import numpy as np
from matplotlib import rcParams
#-------------------------------------------file_opening------------
f1 = pd.read_csv("cvd_final_preprocessed.csv",sep=";")

#------------Check_dataframe------------------------
def check_f1(dataframe, head=10):
    print("\n\n******************* Data Shape *******************\n")
    print("Number of features: ",len(dataframe.columns))
    print("Number of rows: ",len(dataframe))
    print("\n\n*******************Types *******************\n")
    print(dataframe.dtypes)
    print("\n\n******************* Head *******************\n")
    print(dataframe.head(head))
    print("\n\n******************* Tail *******************\n")
    print(dataframe.tail(head))
    print("\n\n******************* NA *******************\n")
    print(dataframe.isnull().sum())
    print("\n\n******************* Quantiles *******************\n")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
print(check_f1(f1))
#--------------------Operation---------------------------------------
#f1.drop(['id'] ,inplace=True,index=None, axis=1)
#f1["age"]=round(f1["age"]/365,0)

print(f1.head())
f1.loc[(f1["age"] < 18), "AGE_Group"] = "Young"
f1.loc[(f1["age"] > 18) & (f1["age"] < 56), "AGE_Group"] = "Mature"
f1.loc[(f1["age"] >= 56), "AGE_Group"] = "Old"


#print(f1.head())
weight = f1["weight"]
height = f1["height"] / 100
f1["BMI"] = round((weight) / (height)**2,2)

#print(f1.head())
f1.loc[(f1["BMI"] < 18.5), "NEW_BMI"] = "under"
f1.loc[(f1["BMI"] >= 18.5) & (f1["BMI"] <= 24.99) ,"NEW_BMI"] = "healthy"
f1.loc[(f1["BMI"] >= 25) & (f1["BMI"] <= 29.99) ,"NEW_BMI"]= "over"
f1.loc[(f1["BMI"] >= 30), "NEW_BMI"] = "obese"

f1.loc[(f1["ap_lo"])<=89, "BP"] = "normal"
f1.loc[(f1["ap_lo"])>=90, "BP"] = "hyper"
f1.loc[(f1["ap_hi"])<=120, "BP"] = "normal"
f1.loc[(f1["ap_hi"])>120, "BP"] = "normal"
f1.loc[(f1["ap_hi"])>=140, "BP"] = "hyper"

#print(f1.head())
f1['AGE_Group'] = f1['AGE_Group'].replace(['Mature','Old'],['0','1'])  #mature=0, old=1
f1['NEW_BMI'] = f1['NEW_BMI'].replace(['healthy','obese','over','under'],['0','1','2','3'])  #healthy=0, obese=1, over=2, under=3
f1['BP'] = f1['BP'].replace(['normal','hyper',],['0','1'])  #normal=0, hyper=1
#print(f1)

pd.DataFrame(f1).to_csv("cvd_dataset.csv", sep=",",   index=False)
#b=f1.groupby('age')['cardio'].mean()
#print(b)
#f1.head()
