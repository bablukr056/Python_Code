from sklearn.ensemble import RnadomForestRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#-------------------Read the data----------------------------------------------
dat = pd.read_csv("testsvm.csv")
X = dat.drop('X', axis=1) 
y = dat['X'].values.tolist()      

X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.25)

print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]))

regressor=RandomForestRegressor(n_estimators=20, random_stae=0)

regressor.fit(X_trn,Ytrn)

y_pred=regressor.predict(X_test)
