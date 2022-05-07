import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,KFold,ShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score
#-------------------------Data-----------------------------------------------------
print("-------------------Data_Set_discription----------------------------------------------","\n")
dat = pd.read_csv("10000.csv")
X = dat.drop('Class', axis=1)
y = dat['Class'].values.tolist()
print("Dimensions of Dataset: ",dat.shape)
X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.20)     #split datasets into training and test.
cv = ShuffleSplit(n_splits=50, random_state=0)                              #split training dataset into small subsets for cross validation.
print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]),"\n")
print("The dimension of the test data Rows: %d Features: %d"%(X_test.shape[0],X_test.shape[1]),"\n")

#---------------------------------------------------------------------------------------------

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234),
               GaussianNB(),
               KNeighborsClassifier(),
               DecisionTreeClassifier(random_state=1234),
               RandomForestClassifier(random_state=1234)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_trn, Y_trn)
    yproba = model.predict_proba(X_test)[::,1]

    fpr, tpr, _ = roc_curve(Y_test,  yproba)
    auc = roc_auc_score(Y_test, yproba)

    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr,
                                        'tpr':tpr,
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
