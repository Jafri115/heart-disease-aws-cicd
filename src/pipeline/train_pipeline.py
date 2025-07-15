# train_pipeline.py

import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transfromation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exceptions import CustomException
from src.logger import logging


def run_training() -> None:
    """Run the end-to-end training pipeline."""
    try:
        logging.info("Starting data ingestion")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        logging.info("Starting data transformation")
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path, test_path
        )

        logging.info("Starting model training")
        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Training completed with F1 score: {score}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training()