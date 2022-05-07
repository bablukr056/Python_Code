import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif,SelectKBest,chi2
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
#-----------------------------------------------------------------
df = pd.read_csv('combined.csv', sep=",")
X = df.drop('Class', axis=1)
y = df['Class'].values.tolist()
print("Dimensions of Dataset: ",df.shape)
X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.10)     #split datasets into training and test.
print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]),"\n")
print("The dimension of the test data Rows: %d Features: %d"%(X_test.shape[0],X_test.shape[1]),"\n")
model = ExtraTreesClassifier()
model.fit(X_trn,Y_trn)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(13).plot(kind='barh')
plt.show()

