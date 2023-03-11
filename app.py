# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:55:58 2023

@author: Akash Dhude
"""
from flask import Flask, request
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
pickle_in = open("classifier.pkl", "rb") # Opening the pickle file
classifier = pickle.load(pickle_in) # Loading pickle file

@app.route("/")
def welcome():
    return "Welcome All"

@app.route("/predict") # Default method is ["GET"]
def predict_note_authentiction():
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted values are " + str(prediction)

@app.route("/predict_file", methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted values for the csv are " + str(list(prediction))



if __name__=="__main__":
    app.run()
