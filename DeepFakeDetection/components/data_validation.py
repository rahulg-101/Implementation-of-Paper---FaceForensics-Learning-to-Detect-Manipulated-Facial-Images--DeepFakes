import os,sys
from DeepFakeDetection.exception import CustomException
from DeepFakeDetection.logger import logging

from DeepFakeDetection.entity.config_entity import DataValidationConfig
from DeepFakeDetection.entity.artifacts_entity import (DataIngestionArtifact,DataValidationArtifact)


class DataValidation():
    def __init__(self,
                data_validation_config = DataValidationConfig,
                data_ingestion_artifact = DataIngestionArtifact):
        
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise CustomException(e,sys)

    def validate_all_existing_files(self)->bool:
        logging.info(f"Trying to Validate whether we have received all files or not !!")
        try:
            
            validation_status = None

            all_files = os.listdir(os.path.join(self.data_ingestion_artifact.data_ingestion_file_path,'data'))
            
            for file in all_files:
                if file not in self.data_validation_config.required_file_list:
                    print(file)
                    validation_status = False
                    os.makedirs(self.data_validation_config.data_validation_dir_name,exist_ok = True)
                    with open(self.data_validation_config.valid_status_file_dir,"w") as f:
                        f.write(f"Validation Status : {validation_status}")

                else:
                    validation_status = True
                    os.makedirs(self.data_validation_config.data_validation_dir_name,exist_ok = True)
                    with open(self.data_validation_config.valid_status_file_dir,"w") as f:
                        f.write(f"Validation Status : {validation_status}")

            return validation_status
        
        except Exception as e:
            raise CustomException(e,sys)

    def initialize_data_validation(self)->DataValidationArtifact:
        logging.info("Entered initiate_data_validation method of DataValidation class")

        try:
            validation_status = self.validate_all_existing_files()
            data_validation_artifact = DataValidationArtifact(validation_status=validation_status)

            logging.info("Exited initiate_data_validation method of DataValidation class")
            logging.info(f"Data Validation artifacts : {data_validation_artifact}")

            return data_validation_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        



    
                



