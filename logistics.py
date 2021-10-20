from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#-------------------Read the data----------------------------------------------
dat = pd.read_csv("mergedsvmfileceRNAepigenetic")
X = dat.drop('X', axis=1) 
y = dat['X'].values.tolist()      

X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.05)
lrc = LogisticRegression(max_iter=1000).fit(X, y)
score = lrc.score(X_test, Y_test)
print(score)
knn = KNeighborsClassifier()
knn.fit(X_trn, Y_trn)
score = knn.score(X_test, Y_test)
print(score)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(300, 3), random_state=1)
clf.fit(X, y)
score = knn.score(X_test, Y_test)
print(score)
