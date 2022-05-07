import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, model_selection
import lazypredict
from lazypredict.Supervised import LazyClassifier
#-----------------------------------------------------------------------------------------------------
dat = pd.read_csv("cvd_healthy_normalize.csv")
X = dat.drop('Class', axis=1)
y = dat['Class'].values.tolist()

X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.10)
print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]))

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_trn, X_test, Y_trn, Y_test)

print(models)
