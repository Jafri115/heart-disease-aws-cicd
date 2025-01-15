
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exceptions import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        '''
        This function will return the preprocessor object
        '''
        try:
            numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
            
            num_pipeline = Pipeline(
                steps = [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())                
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                            steps=[
                            ('imputer',SimpleImputer(strategy='most_frequent')),
                            ('ordinal_encoder',OneHotEncoder()),
                            ('scaler',StandardScaler(with_mean=False))
                            ]
                        )

            preprocessor = ColumnTransformer(
                            [
                            ('num_pipeline',num_pipeline,numeric_features),
                            ('cat_pipeline',cat_pipeline,categorical_features)
                            ]
                        )
            
            return preprocessor
            
        except Exception as e:
            logging.error("Error occured while initializing the DataTransformation class")
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self,train_path,test_path):
        '''
        This function will transform(imputation, standarization, encoding) the data and save the preprocessor object
        '''
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation_obj()
            
            target_column_name="HeartDisease"

            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("Transforming train data")
            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr =preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]   
            
            save_object(self.transformation_config.preprocessor_obj_file_path,preprocessing_obj)
            
            return train_arr,test_arr,self.transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            raise CustomException(e,sys) from e