#importing libraries
import joblib as joblib
import inputScript

#load the pickle file
#classifier = joblib.load("C:\\Users\sriro\Desktop\detecting-phishing-websites-master\\final_models\\xgb_final.pkl")
classifier = joblib.load(r"D:\Projects\ISAA\Phishing website detector\final_models\xgb_final.pkl")

#input url
print("Enter URL:")
url = input()

#checking and predicting
checkprediction = inputScript.main(url)
prediction = classifier.predict(checkprediction)
if prediction==1:
    print("It is Phishing Website!")
else:
    print("It is not Phishing Website!")