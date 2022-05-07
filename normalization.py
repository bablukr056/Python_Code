#----------Import library----------------------------------------
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#----------------------------Read_-------------------------------
Data = pd.read_csv("healthy_all_variables.tsv", sep="\t")
headerList=(Data.columns[:])
print(headerList)
##------------------__delete__target-------------------------------
Data.drop(['cardio'],inplace=True,index=None, axis=1)
headerList=(Data.columns[:])combined.csv
#pd.DataFrame(Data).to_csv("cvd_withoutcat.csv", header=headerList,  index=False)
print(headerList)
###------------------Normalization-----------------
scaler = preprocessing.StandardScaler().fit(Data)
Data=scaler.transform(Data)
#-------------------------Addition_of_Target_Class-------------------------------------------------------------------------
df = pd.DataFrame(Data)
df['Class']=0

###--------------------write_output_result-------------------------------------------------------------------------------
pd.DataFrame(df).to_csv("healthy_normalize.csv",   index=False)
