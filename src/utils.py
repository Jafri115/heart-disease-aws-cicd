# utils.py
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.logger import logging
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
            print(f"Training {model_name}...")
            logging.info(f"Training {model_name}...")
            model = list(models.values())[i]
            para = param[model_name]
            cv = StratifiedKFold(n_splits=3)
            
            def report_nan_scores(cv_results):
                for i, score in enumerate(cv_results['mean_test_score']):
                    if np.isnan(score):
                        print(f"NaN score for parameters: {cv_results['params'][i]}")

            # Hyperparameter tuning
            gs = GridSearchCV(model, para, cv=cv, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            report_nan_scores(gs.cv_results_)

            # Set the best parameters and refit the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            test_model_score = f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
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
    
   