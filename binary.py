import pandas as pd
import numpy as np

df=pd.read_csv("/home/suriya/Subash/ML/loc_allele.tsv",sep='\t')

# d = pd.get_dummies(df.set_index('Location')['Allele'].astype(str)).max(level=0).sort_index()

df["value"]=1
d=pd.pivot_table(df, values="value", index=["Location"], columns="Allele", fill_value=0)


d.to_csv('data.csv',sep="\t")
