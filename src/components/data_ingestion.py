# data_ingestion.py
import os
import sys
from src.components.data_transfromation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")
@dataclass
class DataIngestionConfig:
    '''
    DataIngestionConfig: A class for holding the configuration for data ingestion
    '''
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        '''
        This function will load the raw data and split it into train and test data
        '''
        try:
            pass
            raw_data =  pd.read_csv('notebook/data/heart.csv')
            logging.info("Data loaded successfully")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            raw_data.to_csv(self.ingestion_config.train_data_path, index=False)
            
            logging.info('Splitting the data into train and test')
            
            train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info('Data Ingestion completed successfully')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error("Error occured while initializing the DataIngestion class")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, preproceesor_path = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    
    logging.info(f"Train data path: {train_data_path}")
    logging.info(f"Test data path: {test_data_path}")
    logging.info(f"Preprocessor path: {preproceesor_path}")