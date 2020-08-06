import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from MyPerceptron import MyPerceptron

data=pd.read_csv('student_records.csv')

X=data.iloc[:,0:-1].values
y=data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


# Sklearn's Perceptron -------------------
#clf=Perceptron()

#clf.fit(X_train, y_train)

#y_pred=clf.predict(X_test)

#print(accuracy_score(y_test,y_pred))

# ---------------------------------------

print(X_train.shape)
print(y_train.shape)

clf = MyPerceptron()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))



