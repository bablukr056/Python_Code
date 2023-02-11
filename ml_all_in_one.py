"""
This is a python code for a machine learning experiment using several classifiers to predict a binary classification problem. The classifiers used in the experiment are:

RandomForestClassifier
Logistic Regression
K-Nearest Neighbors (KNN)
Multi-layer Perceptron (MLP)
Decision Tree
Naive Bayes
The code starts by importing the necessary libraries and then reads the dataset from a .csv file using the pandas library.
The data is then split into a training set and a test set using the train_test_split method from the scikit-learn library. 
The training set is further split into smaller subsets for cross-validation using the ShuffleSplit method.

The code then trains and tests the performance of each of the classifiers using the training and test sets. 
For each classifier, the code calculates accuracy, sensitivity, specificity, precision, recall, and F1-score. 
These metrics are used to evaluate the performance of each classifier. Finally, the code plots the confusion matrix and the ROC curve for each classifier to give a graphical representation of the performance.
"""
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
from sklearn.metrics import classification_report, plot_roc_curve,plot_confusion_matrix
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
dat = pd.read_csv("cvd_healthy_normalize.csv")
X = dat.drop('Class', axis=1)
y = dat['Class'].values.tolist()
print("Dimensions of Dataset: ",dat.shape)
X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.20)     #split datasets into training and test.
cv = ShuffleSplit(n_splits=50, random_state=0)                              #split training dataset into small subsets for cross validation.
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
plot_confusion_matrix(RFclf, X_test, Y_test)
plt.title("RandomForestClassifier Confusion Matrix",fontsize=18)
knn_disp=metrics.plot_roc_curve(RFclf, X_test, Y_test)
plt.xlabel("False Positive Rate",fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC curve of RandomForestClassifier",fontsize=18)

#------------------------------LogisticRegression--------------------------------------------------------------------------
print("------------------------------LogisticRegression--------------------------------------------------------------------------","\n")
lrc = LogisticRegression(max_iter=10000)
lrc.fit(X_trn, Y_trn)
print('Accuracy of LogisticRegression on training set: %.5f'%(lrc.score(X_trn, Y_trn)))
print('Accuracy of LogisticRegression on test set: %.5f'%(lrc.score(X_test, Y_test)),"\n")

scores = cross_val_score(lrc, X, y, cv=10, scoring='accuracy')
print("LogisticRegression Scores: ",scores)
print("Cross validation average score", scores.mean())

p=lrc.predict(X_test)
metricP(p,Y_test)
print("\n")
plot_confusion_matrix(lrc, X_test, Y_test)
plt.title("LogisticRegression Confusion Matrix",fontsize=18)
knn_disp=metrics.plot_roc_curve(lrc, X_test, Y_test)
plt.xlabel("False Positive Rate",fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC curve of LogisticRegression",fontsize=18)

#-------------KNeighborsClassifier---------------------------
print("------------------------------KNeighborsClassifier--------------------------------------------------------------------------","\n")
knn = KNeighborsClassifier()
knn.fit(X_trn, Y_trn)
print('Accuracy of KNeighborsClassifier on training set: %.5f'%(knn.score(X_trn, Y_trn)))
print('Accuracy of KNeighborsClassifier on test set: %.5f'%(knn.score(X_test, Y_test)),"\n")

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print("KNeighborsClassifier test set Scores: ",scores)
print("Test set cross validation average score", scores.mean())

p=knn.predict(X_test)
metricP(p,Y_test)

#metrics.plot_roc_curve(knn, X_test, Y_test)
#plt.show()

print("\n")
print("------------------------------Naive_bayes--------------------------------------------------------------------------","\n")
gnb = GaussianNB()
gnb.fit(X_trn, Y_trn)

print('Accuracy of Naive_bayes on training set: %.5f'%(gnb.score(X_trn, Y_trn)))
print('Accuracy of Naive_bayes on test set: %.5f'%(gnb.score(X_test, Y_test)),"\n")

scores = cross_val_score(gnb, X, y, cv=10, scoring='accuracy')
print("Naive_bayes test set Scores: ",scores)
print("Test set cross validation average score", scores.mean())

p=gnb.predict(X_test)
metricP(p,Y_test)
print("\n")
plot_confusion_matrix(gnb, X_test, Y_test)
plt.title("Naive_bayes Confusion Matrix",fontsize=18)
knn_disp=metrics.plot_roc_curve(knn, X_test, Y_test)
plt.xlabel("False Positive Rate",fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC curve of Naive_bayes",fontsize=18)

print("------------------------------DecisionTreeClassifier--------------------------------------------------------------------------","\n")
dt = DecisionTreeClassifier()
dt = dt.fit(X_trn,Y_trn)
print('Accuracy of DecisionTreeClassifier on training set: %.5f'%(dt.score(X_trn, Y_trn)))
print('Accuracy of DecisionTreeClassifier on test set: %.5f'%(dt.score(X_test, Y_test)),"\n")

scores = cross_val_score(dt, X, y, cv=10, scoring='accuracy')
print("DecisionTreeClassifier test set Scores: ",scores)
print("Test set cross validation average score", scores.mean())

y = dt.predict(X_test)
metricP(y,Y_test)
print("\n")
plot_confusion_matrix(dt, X_test, Y_test)
plt.title("DecisionTreeClassifier Confusion Matrix",fontsize=18)
knn_disp=metrics.plot_roc_curve(dt, X_test, Y_test)
plt.xlabel("False Positive Rate",fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC curve of DecisionTreeClassifier",fontsize=18)


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
dt_disp = plot_roc_curve(dt, X_test, Y_test,ax=knn_disp.ax_)
RFclf_disp = plot_roc_curve(RFclf, X_test, Y_test, ax=knn_disp.ax_)
lrc_disp = plot_roc_curve(lrc, X_test, Y_test, ax=knn_disp.ax_)
gnb_disp = plot_roc_curve(gnb, X_test, Y_test, ax=knn_disp.ax_)
svcLin1 = plot_roc_curve(svcLin1, X_test, Y_test, ax=knn_disp.ax_)
scvPoly3_disp = plot_roc_curve(svcPoly3, X_test, Y_test,ax=knn_disp.ax_)
svcLin2 = plot_roc_curve(svcLin2, X_test, Y_test, ax=knn_disp.ax_)
knn_disp.figure_.suptitle("ROC curve comparison")
plt.xlabel("False Positive Rate",fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.title("Receiver Operating Comparison",fontsize=20)

plt.show()
