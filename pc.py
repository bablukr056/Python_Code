import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from matplotlib import pyplot


dat = pd.read_csv("withoutcat.csv")
X = dat.drop('Class', axis=1)

y = dat['Class'].values.tolist()
print("Dimensions of Dataset: ",dat.shape)
X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.30)     #split datasets into training and test.
cv = ShuffleSplit(n_splits=20, random_state=0)                              #split training dataset into small subsets for cross validation.
print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]),"\n")
#-------------KNeighborsClassifier---------------------------
print("------------------------------KNeighborsClassifier--------------------------------------------------------------------------","\n")

model = KNeighborsClassifier()
model.fit(X_trn,Y_trn)
results = permutation_importance(model, X_trn, Y_trn, scoring='accuracy')
importance = results.importances_mean
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
