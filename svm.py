import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, model_selection
from sklearn.metrics import classification_report, plot_roc_curve,plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
#------------------------------Matrix-----------------------------------------------
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
dat = pd.read_csv("cvd_healthy_normalize.csv")
X = dat.drop('Class', axis=1)
y = dat['Class'].values.tolist()

X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.20)

print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]))
#------------------------------KNeighborsClassifier---------------------------------------------------
knn = KNeighborsClassifier()
knn.fit(X_trn, Y_trn)
print('Accuracy of KNeighborsClassifier on training set: %.5f'%(knn.score(X_trn, Y_trn)))
print('Accuracy of KNeighborsClassifier on test set: %.5f'%(knn.score(X_test, Y_test)),"\n")

p=knn.predict(X_test)
metricP(p,Y_test)
print("\n")
plot_confusion_matrix(knn, X_test, Y_test)
plt.title("KNN Confusion Matrix",fontsize=18)
knn_disp=metrics.plot_roc_curve(knn, X_test, Y_test)
plt.xlabel("False Positive Rate",fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC curve of KNeighborsClassifier",fontsize=18)

#-------------------------------------Linear-----------------------------------------------------
svcLin1 = svm.SVC(kernel='linear', C=1.0, gamma='scale').fit(X_trn, Y_trn)
prediction = svcLin1 .predict(X_test)
print('Accuracy of svcLin1 on training set: %.5f'%(svcLin1.score(X_trn, Y_trn)))
print('Accuracy of svcLin1 on test set: %.5f'%(svcLin1.score(X_test, Y_test)),"\n")
y = svcLin1.predict(X_test)
metricP(y,Y_test)
print("\n")
plot_confusion_matrix(svcLin1, X_test, Y_test)
plt.title("SVM-Linear Confusion Matrix",fontsize=18)
svcLin1=metrics.plot_roc_curve(svcLin1, X_test, Y_test)
plt.xlabel("False Positive Rate",fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC curve of SVM-Linear",fontsize=18)

#-------------------------------------PolyNomial-----------------------------------------------------
svcPoly3 = svm.SVC(kernel='poly', C=1.0, degree=4).fit(X_trn, Y_trn)
prediction = svcPoly3.predict(X_test)
print('Accuracy of svcPoly3 on training set: %.5f'%(svcPoly3.score(X_trn, Y_trn)))
print('Accuracy of svcPoly3 on test set: %.5f'%(svcPoly3.score(X_test, Y_test)),"\n")
y = svcPoly3.predict(X_test)
metricP(y,Y_test)
print("\n")
plot_confusion_matrix(svcPoly3, X_test, Y_test)
plt.title("SVM-PolyNomial Confusion Matrix",fontsize=18)
svcPoly3_disp=metrics.plot_roc_curve(svcPoly3, X_test, Y_test)
plt.xlabel("False Positive Rate",fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC curve of SVM-PolyNomial",fontsize=18)

#---------------------------------svcRBF-------------------------------------------------------------
svcRBF = svm.SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_trn, Y_trn)
prediction = svcRBF.predict(X_test)
print('Accuracy of svcRBF on training set: %.5f'%(svcRBF.score(X_trn, Y_trn)))
print('Accuracy of svcRBF on test set: %.5f'%(svcRBF.score(X_test, Y_test)),"\n")
y = svcRBF.predict(X_test)
metricP(y,Y_test)
print("\n")
plot_confusion_matrix(svcRBF, X_test, Y_test)
plt.title("SVM-RBF Confusion Matrix",fontsize=18)
svcRBF_disp=metrics.plot_roc_curve(svcRBF, X_test, Y_test)
plt.xlabel("False Positive Rate",fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC curve of SVM-RBF",fontsize=18)
plt.show()
