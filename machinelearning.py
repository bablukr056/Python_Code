#-----------------------------importLibraries-----------------------------------
import sys
import numpy as np
import pandas as pd
#-----------------------Machine_learning_libraries------------------------------
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#-------------------------Split_dataset------------------------------------------
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,KFold,ShuffleSplit
from sklearn import datasets, metrics, model_selection
from sklearn.metrics import classification_report, plot_roc_curve
#---------------Graph_ploat------------------------------------------------------
import matplotlib as mpl 
import matplotlib.pyplot as plt
#-------------------Confusion matrix---------------------------------------------
def metricP (pred,actual):
   TP = TN = FP = FN = 0
   for i in range(0,len(actual),1):
      if( (actual[i]==1) and (actual[i]==pred[i]) ):
         TP+=1
      if( (actual[i]==0) and (actual[i]==pred[i]) ):
         TN+=1
      if( (actual[i]==1) and (actual[i]!=pred[i]) ):
         FN+=1
      if( (actual[i]==0) and (actual[i]!=pred[i]) ):
         FP+=1
   sens = spec =accuracy = precision = recall = 0
   sens = TP/(TP + FN)
   spec = TN/(TN + FP)
   accuracy = (TP + TN) / (TP+TN+FP+FN)
   precision = TP / (TP + FP)
   recall = TP / (TP + FN)
   F1 = 2 * (precision * recall) / (precision + recall)
   print("Tot: ",len(Y_test)," TP: ",TP," TN: ",TN," FN: ",FN," FP: ",FP," Sensitivity: ",sens," Specificity: ",spec," Acc: ",accuracy," Precision: ",precision," Recall: ",recall," F1-score: ",F1)
#-------------------Read the data----------------------------------------------
print("-------------------Data_Set_discription----------------------------------------------","\n")
dat = pd.read_csv("10000.csv")
X = dat.drop('Class', axis=1)
y = dat['Class'].values.tolist()
print("Dimensions of Dataset: ",dat.shape)
X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.10)     #split datasets into training and test.
cv = ShuffleSplit(n_splits=20, random_state=0)                              #split training dataset into small subsets for cross validation.
print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]),"\n")
print("The dimension of the test data Rows: %d Features: %d"%(X_test.shape[0],X_test.shape[1]),"\n")
headerList=(X_test.columns[:])
#pd.DataFrame(X_test).to_csv("testdata.csv", header=headerList,  index=False)

#------------------------RandomForestClassifier---------------------------------

print("------------------------RandomForestClassifier---------------------------------","\n")
RFclf=RandomForestClassifier(n_estimators=100)
RFclf.fit(X_trn,Y_trn)
print('Accuracy of RandomForestClassifier on training set: %.5f'%(RFclf.score(X_trn, Y_trn)))
print('Accuracy of RandomForestClassifier on test set: %.5f'%(RFclf.score(X_test, Y_test)))

scores = cross_val_score(RFclf, X, y, cv=10, scoring='accuracy')
print("RandomForestClassifier Scores: ",scores)
print("Cross validation average score", scores.mean())

p=RFclf.predict(X_test)
metricP(p,Y_test)
print("\n")

#------------------------------LogisticRegression--------------------------------------------------------------------------
print("------------------------------LogisticRegression--------------------------------------------------------------------------","\n")
lrc = LogisticRegression(max_iter=10000)
lrc.fit(X_trn, Y_trn)
print('Accuracy of LogisticRegression on training set: %.5f'%(lrc.score(X_trn, Y_trn)))
print('Accuracy of LogisticRegression on test set: %.5f'%(lrc.score(X_test, Y_test)),"\n")

scores = cross_val_score(lrc, X, y, cv=10, scoring='accuracy')
print("RandomForestClassifier Scores: ",scores)
print("Cross validation average score", scores.mean())

p=lrc.predict(X_test)
metricP(p,Y_test)
print("\n")
#-------------KNeighborsClassifier---------------------------
print("------------------------------KNeighborsClassifier--------------------------------------------------------------------------","\n")
knn = KNeighborsClassifier()
knn.fit(X_trn, Y_trn)
print('Accuracy of KNeighborsClassifier on training set: %.5f'%(knn.score(X_trn, Y_trn)))
print('Accuracy of KNeighborsClassifier on test set: %.5f'%(knn.score(X_test, Y_test)),"\n")

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print("RandomForestClassifier test set Scores: ",scores)
print("Test set cross validation average score", scores.mean())

p=knn.predict(X_test)
metricP(p,Y_test)

#metrics.plot_roc_curve(knn, X_test, Y_test)
#plt.show()

print("\n")

print("------------------------------Support Vector Machine--------------------------------------------------------------------------","\n")
svcLin1 = svm.SVC(kernel='linear', C=1.0, gamma='scale').fit(X_trn, Y_trn)

svcRBF = svm.SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_trn, Y_trn)

svcPoly3 = svm.SVC(kernel='poly', C=1.0, degree=4).fit(X_trn, Y_trn)

svcLin2 = svm.LinearSVC(C=1.0, max_iter=10000).fit(X_trn, Y_trn)

for i, classifier in enumerate((svcLin1, svcRBF, svcPoly3, svcLin2)):
   print('Accuracy of %s on training set: %.5f'%(classifier,classifier.score(X_trn, Y_trn)))
   print('Accuracy of %s on test set: %.5f'%(classifier,classifier.score(X_test, Y_test)))

#------------------------------------ROC_CURVE--------------------------------

knn_disp = plot_roc_curve(knn, X_test, Y_test)

svcLin1 = plot_roc_curve(svcLin1, X_test, Y_test, ax=knn_disp.ax_)
scvPoly3_disp = plot_roc_curve(svcPoly3, X_test, Y_test, ax=knn_disp.ax_)
knn_disp.figure_.suptitle("ROC curve comparison")

plt.show()

