# model_trainer.py
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# Modelling
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from src.exceptions import CustomException
from src.logger import logging


from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training process.")
            logging.info("Splitting training and test input data.")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Defining candidate models and their hyperparameter grids.")
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Support Vector Classifier": SVC(probability=True),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params = {
                # [Your parameter grid remains unchanged...]
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1],
                    'penalty': ['l2'],  # Use only 'l2' for 'lbfgs'
                    'solver': ['lbfgs']
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                "Support Vector Classifier": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'degree': [2, 3, 4]
                },
                "Decision Tree Classifier": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest Classifier": {
                    'n_estimators': [50, 100, 200, 300, 400],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "CatBoost Classifier": {
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'iterations': [50, 100, 200],
                    'l2_leaf_reg': [1, 3, 5, 7]
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }

            }


            logging.info("Starting model evaluation and hyperparameter tuning.")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with F1 score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                logging.warning("No model met the performance threshold.")
                raise CustomException("No best model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Saved best model: {best_model_name} to {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            f1score = f1_score(y_test, predicted)
            logging.info(f"Final F1 score on test data: {f1score:.4f}")

            return f1score

        except Exception as e:
            logging.error("Exception occurred in ModelTrainer", exc_info=True)
            raise CustomException(e, sys)
        

