import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('C:/Backup/E back/Projects/Diamond_Price/Flask/Flask/model.pkl', 'rb'))
#scale = pickle.load(open('C:/Users/SmartbridgePC/Desktop/AIML/Guided projects/rainfall_prediction/IBM flask push/Rainfall IBM deploy/scale.pkl','rb'))

@app.route('/')# route to display the home page
def home():
    return render_template('index.html') #rendering the home page

@app.route('/input')
def pred():
    return render_template('details.html')

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    features_values=[np.array(input_feature)]
    print(features_values)
    names = [[ 'carat','cut', 'color', 'clarity', 'depth', 'table', 'x', 'y',
       'z']]
    data = pandas.DataFrame(features_values,columns=names)

     # predictions using the loaded model file
    prediction=model.predict(data)
    print(prediction[0])
    text = "Hence based on calculation the estimated Price of Diamond (in dollars) is :"
    return render_template("result.html",prediction_text = text + str(prediction[0]))
     # showing the prediction results in a UI
if __name__=="__main__":
        app.run(debug=False, port=5000)
    

 
