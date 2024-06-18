import os,sys
from DeepFakeDetection.logger import logging
from DeepFakeDetection.exception import CustomException

from DeepFakeDetection.entity.config_entity import DataIngestionConfig
from DeepFakeDetection.entity.artifacts_entity import DataIngestionArtifact

import kaggle

kaggle.api.authenticate()

class DataIngestion:
    def __init__(self,data_ingestion_config = DataIngestionConfig()) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
        
    def download_dataset(self):
        """
        Fetch data from URL

        """
        try:
            dataset_name = self.data_ingestion_config.dataset_name
            dataset_path = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(dataset_path,exist_ok=True)
            logging.info(f"Downloading data from kaggle/{dataset_name} into file {dataset_path}")


            kaggle.api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
            logging.info(f"Downloaded data from kaggle/{dataset_name} into file {dataset_path}")

            return dataset_path
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        """
        This function will return dataset_path
        for our artifacts_entity.py file's DataIngestionArtifact class
        """

        logging.info(f"Entered initiate_data_ingestion method of DataIngestion class")

        try:
            dataset_path = self.download_dataset()
            
            data_ingestion_artifact = DataIngestionArtifact(data_ingestion_file_path = dataset_path)
            
            logging.info("Exited initiate_data_ingestion method of DataIngestion classs")
            
            logging.info(f"Data ingestion artifact : {data_ingestion_artifact}")

            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)





