# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 22:09:00 2021

@author: hp
"""



from flask import Flask, render_template, request
from flask import jsonify
import requests
import pickle
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open("F:/Project/Back up files/final files/model.pkl", "rb"))
@app.route('/',methods=['GET'])
def Home():
    return render_template('C:/Users/hp/Deployment master/templates/index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    
    prediction=model.predict(np.array([['date','Hyd_Price']]))
    
    output = round(prediction[0],7)
    return render_template('index.html', prediction_text = 'prediction {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

   

   























    
