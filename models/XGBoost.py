#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import joblib
# starting time
start = time.time()
#importing the dataset
dataset = pd.read_csv("CC:\Users\kusha\Downloads\ISAA Project-20220911T184955Z-001\ISAA Project\models\datasets\phishcoop.csv")
#dataset = pd.read_csv("C:\\Users\sriro\Desktop\detecting-phishing-websites-master\models\datasets\dataset_full.csv")
dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values

#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )

#XGBoost Classification model
from xgboost import XGBClassifier, XGBRegressor
# instantiate the model
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
#fit the model
xgb.fit(x_train, np.ravel(y_train))

#printing best parameters 
y_test_xgb = xgb.predict(x_test)
y_train_xgb = xgb.predict(x_train)

end = time.time()
# total time taken
print(f"Runtime of the program is {end - start}")
#computing the accuracy of the model performance
from sklearn.metrics import accuracy_score,confusion_matrix

cm = confusion_matrix(y_test,y_test_xgb)
acc_test_xgb = accuracy_score(y_test,y_test_xgb)
print(cm)
print("XGBoost : Accuracy on test Data:",acc_test_xgb)

joblib.dump(xgb, "C:\Users\kusha\Downloads\ISAA Project-20220911T184955Z-001\ISAA Project\final_models\xgb_finalphishcoop.pkl")

base_xgb = XGBRegressor(random_state=42)
base_xgb.fit(x_train, np.ravel(y_train))

yhat_xgb = base_xgb.predict(x_test)

s3=round((base_xgb.score(x_test,y_test))*100,3)

print("Score is: ",s3)


names = dataset.iloc[:,:-1].columns
importances =xgb.feature_importances_
sorted_importances = sorted(importances, reverse=True)
indices = np.argsort(-importances)
var_imp = pd.DataFrame(sorted_importances, names[indices], columns=['importance'])

plt.title("Variable Importances")
plt.barh(np.arange(len(names)), sorted_importances, height = 0.7)
plt.yticks(np.arange(len(names)), names[indices], fontsize=7)
plt.xlabel('Relative Importance')
plt.show()
