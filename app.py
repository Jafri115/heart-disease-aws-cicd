# app.py
from flask import Flask, request, render_template

import numpy as np
import pandas as pd
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            Age=request.form.get('age'),
            RestingBP=request.form.get('resting_bp'),
            Cholesterol=request.form.get('cholesterol'),
            FastingBS=request.form.get('fasting_bs'),
            MaxHR=request.form.get('max_hr'),
            Oldpeak=request.form.get('oldpeak'),
            ST_Slope=request.form.get('st_slope'),
            Sex=request.form.get('sex'),
            ChestPainType=request.form.get('chest_pain_type'),
            RestingECG=request.form.get('resting_ecg'),
            ExerciseAngina=request.form.get('exercise_angina')
        )
        
        pred_df = data.get_data_as_frame()
        print(pred_df)
        logging.info(f"Dataframe created successfully")
        
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        prediction_result = 'yes' if int(result[0]) == 1 else 'no'

        
        return render_template('home.html', result=(f'Heart Disease Prediciton : {prediction_result}'))
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
            
    