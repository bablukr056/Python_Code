from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#-------------------Read the data----------------------------------------------
dat = pd.read_csv("tetraNuclNormData.csv")
X = dat.drop('X', axis=1) 
y = dat['X'].values.tolist()      

X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.10)

print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]))

svcLin1 = svm.SVC(kernel='linear', C=1.0, gamma='scale').fit(X_trn, Y_trn)
svcRBF = svm.SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_trn, Y_trn)
svcPoly3 = svm.SVC(kernel='poly', C=1.0, degree=3).fit(X_trn, Y_trn)
svcLin2 = svm.LinearSVC(C=1.0, max_iter=10000).fit(X_trn, Y_trn)

for i, classifier in enumerate((svcLin1, svcRBF, svcPoly3, svcLin2)):
   print('Accuracy of %s on training set: %.5f'%(classifier,classifier.score(X_trn, Y_trn)))
   print('Accuracy of %s on test set: %.5f'%(classifier,classifier.score(X_test, Y_test)))
