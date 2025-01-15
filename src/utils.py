import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException

def save_object(file_path:str,obj):
    '''
    this function will save the object as pickle file
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            # Hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)

            # Set the best parameters and refit the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate the model
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            # Store the test F1-score in the report
            report[model_name] = {
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Test F1-Score': test_f1
            }

        return report

    except Exception as e:
        print(f"Error: {str(e)}")
        return {}
    
def load_object(file_path:str):
    '''
    this function will load the object from pickle file
    '''
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise CustomException(e,sys)
    
   