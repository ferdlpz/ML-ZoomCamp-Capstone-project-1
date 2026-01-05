import pickle

from flask import Flask, request, jsonify
import numpy as np


model_file = 'model_min_samples_leaf=1_max_depth=5_n_estimators=40.bin'

import os
os.chdir('/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-1/04 Script predict')

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    disease = (y_pred >= 0.5).astype(int)

    result = {
        'Heart_Disease_probability': float(y_pred),
        'Heart_Disease': bool(disease)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)