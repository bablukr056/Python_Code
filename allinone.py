import sys
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#-------------------Confusion matrix-------------------------------------------
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
dat = pd.read_csv("Final_norm_data.csv")
X = dat.drop('Class', axis=1) 
y = dat['Class'].values.tolist()      

X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.10)

print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]))
#------------------------RandomForestClassifier---------------------------------
RFclf=RandomForestClassifier(n_estimators=100)
RFclf.fit(X_trn,Y_trn)
print('Accuracy of RandomForestClassifier on training set: %.5f'%(RFclf.score(X_trn, Y_trn)))
print('Accuracy of RandomForestClassifier on test set: %.5f'%(RFclf.score(X_test, Y_test)))
p=RFclf.predict(X_test)
metricP(p,Y_test)
#------------------------------LogisticRegression-----------
lrc = LogisticRegression(max_iter=10000)
lrc.fit(X_trn, Y_trn)
print('Accuracy of LogisticRegression on training set: %.5f'%(lrc.score(X_trn, Y_trn)))
print('Accuracy of LogisticRegression on test set: %.5f'%(lrc.score(X_test, Y_test)))
p=lrc.predict(X_test)
metricP(p,Y_test)

#-------------KNeighborsClassifier---------------------------
knn = KNeighborsClassifier()
knn.fit(X_trn, Y_trn)
print('Accuracy of KNeighborsClassifier on training set: %.5f'%(knn.score(X_trn, Y_trn)))
print('Accuracy of KNeighborsClassifier on test set: %.5f'%(knn.score(X_test, Y_test)))
p=knn.predict(X_test)
metricP(p,Y_test)

#----------MLPClassifier--------------------
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(300, 3), random_state=1)
mlp.fit(X_trn, Y_trn)
print('Accuracy of MLPClassifier on training set: %.5f'%(mlp.score(X_trn, Y_trn)))
print('Accuracy of MLPClassifier on test set: %.5f'%(mlp.score(X_test, Y_test)))
p=mlp.predict(X_test)
metricP(p,Y_test)
#--------------------------Naive_bayes-----------------------
gnb = GaussianNB()
gnb.fit(X_trn, Y_trn)
print('Accuracy of Naive_bayes on training set: %.5f'%(gnb.score(X_trn, Y_trn)))
print('Accuracy of Naive_bayes on test set: %.5f'%(gnb.score(X_test, Y_test)))
p=gnb.predict(X_test)
metricP(p,Y_test)


###------------------------DecisionTreeClassifier----------------------------------
#dt = DecisionTreeClassifier()
#dt = dt.fit(X_trn,Y_trn)
#y_pred = dt.predict(X_test)
#print("Accuracy DecisionTreeClassifier :",metrics.accuracy_score(Y_test, y_pred))
#print('Accuracy of DecisionTreeClassifier on training set: %.5f'%(dt.score(X_trn, Y_trn)))
#print('Accuracy of DecisionTreeClassifier on test set: %.5f'%(dt.score(X_test, Y_test)))
####------------------------SVM----------------------------------------------------
###svcLin1 = svm.SVC(kernel='linear', C=1.0, gamma='scale').fit(X_trn, Y_trn)
###svcRBF = svm.SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_trn, Y_trn)
###svcPoly3 = svm.SVC(kernel='poly', C=1.0, degree=4).fit(X_trn, Y_trn)
###svcLin2 = svm.LinearSVC(C=1.0, max_iter=10000).fit(X_trn, Y_trn)

###for i, classifier in enumerate((svcLin1, svcRBF, svcPoly3, svcLin2)):
   ###print('Accuracy of %s on training set: %.5f'%(classifier,classifier.score(X_trn, Y_trn)))
   ###print('Accuracy of %s on test set: %.5f'%(classifier,classifier.score(X_test, Y_test)))
