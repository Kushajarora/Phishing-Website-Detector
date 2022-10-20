#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time



# starting time
start = time.time()

#importing the dataset
#dataset = pd.read_csv("C:\\Users\sriro\Desktop\detecting-phishing-websites-master\models\datasets\dataset_full.csv")
dataset = pd.read_csv("C:\Users\kusha\Downloads\ISAA Project-20220911T184955Z-001\ISAA Project\models\datasets\phishcoop.csv")
dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values

#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )

from lightgbm import LGBMClassifier
model = LGBMClassifier()
model.fit(x_train, np.ravel(y_train))
 
# Predicting the Target variable
pred = model.predict(x_test)
# end time

from sklearn.metrics import confusion_matrix
accuracy = model.score(x_test, y_test)
cm = confusion_matrix(y_test,pred)
print(accuracy)
print(cm)


from lightgbm import LGBMRegressor


base_lgbmr = LGBMRegressor(random_state=42)
base_lgbmr.fit(x_train, np.ravel(y_train))

yhat_lgbmr = base_lgbmr.predict(x_test)

s3=round((base_lgbmr.score(x_test,y_test))*100,3)

print("Score is: ",s3)

end = time.time()
# total time taken
print(f"Runtime of the program is {end - start}")

names = dataset.iloc[:,:-1].columns
importances =model.feature_importances_
sorted_importances = sorted(importances, reverse=True)
indices = np.argsort(-importances)
var_imp = pd.DataFrame(sorted_importances, names[indices], columns=['importance'])

plt.title("Variable Importances")
plt.barh(np.arange(len(names)), sorted_importances, height = 0.7)
plt.yticks(np.arange(len(names)), names[indices], fontsize=7)
plt.xlabel('Relative Importance')
plt.show()