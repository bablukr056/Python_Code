import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
#-----------------------------------------------------------------
df = pd.read_csv('10000.csv', sep=",")

X=df.drop('Class',1)
y=df['Class']

print("Dimensions of Dataset: ",df.shape)
X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.30)     #split datasets into training and test.
#cv = ShuffleSplit(n_splits=20, random_state=0)                              #split training dataset into small subsets for cross validation.
print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]),"\n")
print("The dimension of the test data Rows: %d Features: %d"%(X_test.shape[0],X_test.shape[1]),"\n")


