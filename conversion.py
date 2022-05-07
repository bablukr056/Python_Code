import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#-----------------------------------
f1 = pd.read_csv("test.csv",sep="\t")

#headerList=(Data.columns[1:])
#------------conversion into 0,1,2,3--------------
f1['AGE_Group'] = f1['AGE_Group'].replace(['Mature','Old'],['0','1'])  #mature=0, old=1
f1['NEW_BMI'] = f1['NEW_BMI'].replace(['healthy','obese','over','under'],['0','1','2','3'])  #healthy=0, obese=1, over=2, under=3
f1['BLOOD_PRESSURE'] = f1['BLOOD_PRESSURE'].replace(['normal','hyper',],['0','1'])  #normal=0, hyper=1
#---------------------------------------------------------
#print(f1.head(50))


pd.DataFrame(f1).to_csv("bablu.csv",sep=",", index=False)
