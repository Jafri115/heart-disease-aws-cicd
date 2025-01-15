import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Age: int,
        RestingBP: int,
        Cholesterol: int,
        FastingBS: int,
        MaxHR: int,
        Oldpeak: float,
        ST_slope: str,
        Sex: str,
        ChestPainType: str,
        RestingECG: str,
        ExcersiceAngina: str):
        self.age = Age
        self.resting_bp = RestingBP
        self.cholesterol = Cholesterol
        self.fasting_bs = FastingBS
        self.max_hr = MaxHR
        self.oldpeak = Oldpeak
        self.st_slope = ST_slope
        self.sex = Sex
        self.chest_pain_type = ChestPainType
        self.resting_ecg = RestingECG
        self.exercise_angina = ExcersiceAngina

        
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "resting_bp": [self.resting_bp],
                "cholesterol": [self.cholesterol],
                "fasting_bs": [self.fasting_bs],
                "max_hr": [self.max_hr],
                "oldpeak": [self.oldpeak],
                "st_slope": [self.st_slope],
                "sex":[self.sex],
                "chest_pain_type": [self.chest_pain_type],
                "resting_ecg": [self.resting_ecg],
                "exercise_angina": [self.exercise_angina],

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
