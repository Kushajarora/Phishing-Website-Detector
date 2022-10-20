import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import joblib

#importing the dataset
dataset = pd.read_csv("C:\Users\kusha\Downloads\ISAA Project-20220911T184955Z-001\ISAA Project\models\datasets\phishcoop.csv")
dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values

#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )
classifier = SVC(C=1000, kernel = 'rbf', gamma = 0.2 , random_state = 0)
classifier.fit(x_train, np.ravel(y_train))

y_pred = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_test = accuracy_score(y_test,y_pred)
print("Best Accuracy: ",acc_test)
print(cm)

#pickle file joblib
joblib.dump(classifier, 'C:\Users\kusha\Downloads\ISAA Project-20220911T184955Z-001\ISAA Project\final_models\svm_final.pkl')