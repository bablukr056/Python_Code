#-----------------------------importLibraries-----------------------------------
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
#-----------------------Machine_learning_libraries------------------------------
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
import matplotlib.pyplot as plt
#-------------------------Split_dataset------------------------------------------
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,KFold,ShuffleSplit
from sklearn import datasets, metrics, model_selection
from sklearn.metrics import classification_report, plot_roc_curve,plot_confusion_matrix,mean_absolute_error,accuracy_score
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

print("-------------------Data_Set_discription----------------------------------------------","\n")
dat = pd.read_csv("cvd_healthy_normalize.csv")
X = dat.drop('Class', axis=1)
y = dat['Class'].values.tolist()

print("Dimensions of Dataset: ",dat.shape)
X_trn, X_test, Y_trn, Y_test = train_test_split(X, y, test_size = 0.30)     #split datasets into training and test.
cv = ShuffleSplit(n_splits=20, random_state=0)                              #split training dataset into small subsets for cross validation.
print("The dimension of the training data Rows: %d Features: %d"%(X_trn.shape[0],X_trn.shape[1]),"\n")

print("------------------------------KNeighborsClassifier--------------------------------------------------------------------------","\n")
knn = KNeighborsClassifier()
knn.fit(X_trn, Y_trn)
print('Accuracy of KNeighborsClassifier on training set: %.5f'%(knn.score(X_trn, Y_trn)))
print('Accuracy of KNeighborsClassifier on test set: %.5f'%(knn.score(X_test, Y_test)),"\n")

p=knn.predict(X_test)
metricP(p,Y_test)
print("\n")
scores = cross_val_score(knn, X, y, cv=50, scoring='accuracy')
print("Cross validation average score", scores.mean(),"\n")
#--------------------K_Value---------------------------------
Range_k = range(1,15)
scores = {}
scores_list = []
for k in Range_k:
   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(X_trn, Y_trn)
   y_pred = classifier.predict(X_test)
   scores[k] = metrics.accuracy_score(Y_test,y_pred)
   scores_list.append(metrics.accuracy_score(Y_test,y_pred))

result1 = metrics.classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
plt.plot(Range_k,scores_list)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("Optimal value of K", fontsize=18)
plt.show()
#---------------plot_confusion_matrix---------------------------------
plot_confusion_matrix(knn, X_test, Y_test)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion matrix of KNN Classifier', fontsize=18)
plt.show()
#-----------------------ROC-------------------------------------
metrics.plot_roc_curve(knn, X_test, Y_test)
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC curve of KNeighbors Classifier', fontsize=18)
plt.show()
#-------------------------classification_report-----------------
result1 = classification_report(Y_test, p)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,p)
print("Accuracy:",result2)
