import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
#-----------------------------------------------------------------

df = pd.read_csv('cvd_healthy.csv', sep=",")

X=df.drop('cardio',1)
y=df['cardio']

X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=1)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')
