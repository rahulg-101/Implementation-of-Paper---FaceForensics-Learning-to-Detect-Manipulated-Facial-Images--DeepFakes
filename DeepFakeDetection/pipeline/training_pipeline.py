import os,sys
from DeepFakeDetection.exception import CustomException
from DeepFakeDetection.logger import logging

from DeepFakeDetection.components.data_ingestion import DataIngestion
from DeepFakeDetection.components.data_validation import DataValidation
from DeepFakeDetection.entity.config_entity import (DataIngestionConfig,DataValidationConfig)
from DeepFakeDetection.entity.artifacts_entity import (DataIngestionArtifact,DataValidationArtifact)

class TrainingPipeline():
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info("Entered the start_data_ingestion method of TrainingPipeline class")
            logging.info(f"Getting the data from Kaggle Datasets {self.data_ingestion_config.dataset_name}")

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"Got the data from Kaggle Datasets {self.data_ingestion_config.dataset_name}")
            logging.info("Exited the start_data_ingestion method of TrainingPipeline class")

            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info("Entered the start_data_validation method of TrainingPipeline class")
            logging.info(f"Validating the number of files that we have in {data_ingestion_artifact.data_ingestion_file_path}")

            data_validation = DataValidation(data_ingestion_artifact=self.data_ingestion_config,
                                             data_validation_config =self.data_validation_config)

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed Data Validation operation")

            logging.info("Exited the start_data_validation method of TrainPipeline class")

            return data_validation_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        






    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact)
            if data_validation_artifact.validation_status == True:
                return None         # Temporarily added None 


        except Exception as e:
            raise CustomException(e,sys)
        
        

