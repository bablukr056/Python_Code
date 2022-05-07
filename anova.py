import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

df = pd.read_csv('combined.csv', sep=",")
#X = df.drop('Class', axis=1)
#y = df['Class'].values.tolist()

print("Dimensions of Dataset: ",df.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("The dimension of the training data Rows: %d Features: %d"%(X_train.shape[0],X_train.shape[1]),"\n")
print("The dimension of the test data Rows: %d Features: %d"%(X_test.shape[0],X_test.shape[1]),"\n")

anova_filter = SelectKBest(f_classif, k=3)
clf = LinearSVC()
anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)
y_pred = anova_svm.predict(X_test)
print(classification_report(y_test, y_pred))

