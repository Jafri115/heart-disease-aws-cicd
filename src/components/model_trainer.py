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
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
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
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1],
                    'penalty': ['l2'],  # Use only 'l2' for 'lbfgs'
                    'solver': ['lbfgs']
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],  # Number of neighbors
                    'weights': ['uniform', 'distance'],  # Weight function
                    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
                },
                "Support Vector Classifier": {
                    'C': [0.1, 1, 10, 100],  # Regularization parameter
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
                    'gamma': ['scale', 'auto'],  # Kernel coefficient
                    'degree': [2, 3, 4]  # Degree for 'poly' kernel
                },
                "Decision Tree Classifier": {
                    'criterion': ['gini', 'entropy', 'log_loss'],  # Splitting criterion
                    'max_depth': [None, 10, 20, 30, 40, 50],  # Depth of the tree
                    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
                    'min_samples_leaf': [1, 2, 4]  # Minimum samples in a leaf node
                },
                "Random Forest Classifier": {
                    'n_estimators': [50, 100, 200, 300, 400],  # Number of trees
                    'criterion': ['gini', 'entropy'],  # Splitting criterion
                    'max_features': ['sqrt', 'log2', None],  # Features to consider at split
                    'max_depth': [None, 10, 20, 30],  # Depth of each tree
                    'min_samples_split': [2, 5, 10]  # Minimum samples to split
                },
                "CatBoost Classifier": {
                    'depth': [4, 6, 8, 10],  # Depth of the tree
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
                    'iterations': [50, 100, 200],  # Number of boosting iterations
                    'l2_leaf_reg': [1, 3, 5, 7]  # L2 regularization
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 200],  # Number of weak learners
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],  # Step size
                    'algorithm': ['SAMME', 'SAMME.R']  # Boosting algorithm
                }

            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            f1score = f1_score(y_test,predicted)
            return f1score
            
        except Exception as e:
            raise CustomException(e,sys)