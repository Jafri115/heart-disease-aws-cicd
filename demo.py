# demo.py
import gradio as gr
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging

# Define prediction function
def predict_heart_disease(age, sex, chest_pain_type, resting_bp, cholesterol, 
                          fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    try:
        # Create data object
        data = CustomData(
            Age=age,
            Sex=sex,
            ChestPainType=chest_pain_type,
            RestingBP=resting_bp,
            Cholesterol=cholesterol,
            FastingBS=fasting_bs,
            RestingECG=resting_ecg,
            MaxHR=max_hr,
            ExerciseAngina=exercise_angina,
            Oldpeak=oldpeak,
            ST_Slope=st_slope
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_frame()
        logging.info(f"Input DataFrame: \n{pred_df}")

        # Make prediction
        pipeline = PredictPipeline()
        result = pipeline.predict(pred_df)
        prediction_result = 'Yes' if int(result[0]) == 1 else 'No'

        return f"Heart Disease Prediction: {prediction_result}"
    
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        return "Error during prediction."

# Define Gradio interface
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["M", "F"], label="Sex"),
        gr.Dropdown(["TA", "ATA", "NAP", "ASY"], label="Chest Pain Type"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Dropdown([0, 1], label="Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)"),
        gr.Dropdown(["Normal", "ST", "LVH"], label="Resting ECG"),
        gr.Number(label="Max Heart Rate Achieved"),
        gr.Dropdown(["Y", "N"], label="Exercise Induced Angina"),
        gr.Number(label="Oldpeak"),
        gr.Dropdown(["Up", "Flat", "Down"], label="ST Slope")
    ],
    outputs="text",
    title="Heart Disease Prediction App",
    description="Enter patient information to predict the presence of heart disease."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
