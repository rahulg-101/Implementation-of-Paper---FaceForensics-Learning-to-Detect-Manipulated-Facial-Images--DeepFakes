import os,sys
import json
from DeepFakeDetection.exception import CustomException
from DeepFakeDetection.logger import logging

from DeepFakeDetection.components.data_ingestion import DataIngestion
from DeepFakeDetection.components.data_validation import DataValidation
from DeepFakeDetection.components.data_transformation import DataTransform
from DeepFakeDetection.components.model_trainer import ModelTrainer

from DeepFakeDetection.entity.config_entity import (DataIngestionConfig,DataValidationConfig,DataTransformConfig,ModelTrainerConfig)
from DeepFakeDetection.entity.artifacts_entity import (DataIngestionArtifact,DataValidationArtifact,DataTransformArtifact,ModelTrainerArtifacts)
from DeepFakeDetection.utils.main_utils import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

class TrainingPipeline():
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transform_config = DataTransformConfig()
        self.model_trainer_config = ModelTrainerConfig()

    
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

            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config =self.data_validation_config)

            data_validation_artifact = data_validation.initialize_data_validation()

            logging.info("Performed Data Validation operation")

            logging.info("Exited the start_data_validation method of TrainPipeline class")

            return data_validation_artifact 
        
        except Exception as e:
            raise CustomException(e,sys)
        
    

    def start_data_transformation_generation(self,data_ingestion_artifact:DataIngestionArtifact):
        logging.info("Entered the start_data_transformation_generation method of TrainingPipeline class")
        
        try:
            data_transform = DataTransform(
                data_transform_config = self.data_transform_config,
                data_ingestion_artifact = data_ingestion_artifact
            )
            data_transform.initialize_config_save()

            
            train_datagen_params = load_config('train.json', self.data_transform_config.data_generator_dir)
            # train_datagen = ImageDataGenerator(**train_datagen_params)

            val_datagen_params = load_config('val.json', self.data_transform_config.data_generator_dir)
            # val_datagen = ImageDataGenerator(**val_datagen_params)

            test_datagen_params = load_config('test.json', self.data_transform_config.data_generator_dir)
            # test_datagen = ImageDataGenerator(**test_datagen_params)
            
            logging.info("ImageDataGenerator objects have been created")
            
            # train_generator = train_datagen.flow_from_directory(**data_transform.create_data_generator_config('train'))
            # val_generator = val_datagen.flow_from_directory(**data_transform.create_data_generator_config('val'))
            # test_generator = test_datagen.flow_from_directory(**data_transform.create_data_generator_config('test'))

            generator_config = {
            'train': data_transform.create_data_generator_config('train'),
            'val': data_transform.create_data_generator_config('val'),
            'test': data_transform.create_data_generator_config('test'),
            'train_datagen_params': train_datagen_params,
            'val_datagen_params': val_datagen_params,
            'test_datagen_params': test_datagen_params
                                }
            logging.info("Data Subset generators have been successfully created")
            logging.info("Exited the start_data_transformation_generation method of TrainingPipeline class")
            
            return generator_config
                 
        except Exception as e:
            raise CustomException(e,sys)


    def start_model_trainer(self, generator_config):
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                generator_config=generator_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)


    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact)

            if data_validation_artifact.validation_status:
                generator_config = self.start_data_transformation_generation(
                    data_ingestion_artifact=data_ingestion_artifact)
                
                model_trainer_artifact = self.start_model_trainer(generator_config=generator_config)

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    obj = TrainingPipeline()
    obj.run_pipeline()