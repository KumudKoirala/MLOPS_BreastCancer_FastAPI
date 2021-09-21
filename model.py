from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import joblib
cancer=load_breast_cancer()
X=cancer.data
y=cancer.target
X_train,X_test,y_train,y_test=train_test_split(X,y)
model =DecisionTreeClassifier()
model.fit(X,y)
joblib.dump(model,"breast_cancer_classifier.joblib")
