#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and preprocessors
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        engine_size = float(data['engine_size'])
        cylinders = int(data['cylinders'])
        fuel_city = float(data['fuel_city'])
        fuel_hwy = float(data['fuel_hwy'])
        fuel_comb = float(data['fuel_comb'])
        transmission = data['transmission']
        fuel_type = data['fuel_type']
        vehicle_class = data['vehicle_class']
        
        # Prepare numerical data
        X_num = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb]])

        # Prepare categorical data
        X_cat = pd.DataFrame([[transmission, fuel_type, vehicle_class]], 
                              columns=['transmission', 'fuel_type', 'vehicle_class'])
        
        # Apply preprocessing
        X_num_scaled = scaler.transform(X_num)  # Scale numerical features
        X_cat_encoded = encoder.transform(X_cat).toarray()  # Encode categorical features
        
        # Combine numerical + categorical features
        X_processed = np.hstack((X_num_scaled, X_cat_encoded))

        # Make prediction
        prediction = best_model.predict(X_processed)[0]
        
        return jsonify({"CO2 Emission Prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

