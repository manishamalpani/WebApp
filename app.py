import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load pre-trained model (you can train the model beforehand and save it using joblib)
# For the Titanic dataset, a simple model is used
model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Fetch data from the form
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])

        # Map categorical variable 'sex' to numeric (1 for male, 0 for female)
        sex = 1 if sex == 'male' else 0

        # Prepare data for prediction
        data = np.array([[pclass, sex, age, sibsp, parch, fare]])
        
        # Make prediction
        prediction = model.predict(data)
        prediction_text = 'survive' if prediction[0] == 1 else 'not survive'
        
        return render_template('index.html', prediction_text=f'The passenger will {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
