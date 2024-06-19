import os,sys
import json
from DeepFakeDetection.exception import CustomException
from DeepFakeDetection.logger import logging

from DeepFakeDetection.entity.artifacts_entity import (DataIngestionArtifact,DataTransformArtifact)
from DeepFakeDetection.entity.config_entity import DataTransformConfig
from DeepFakeDetection.utils.main_utils import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


class DataTransform():
    def __init__(self,
                 data_transform_config = DataTransformConfig,
                 data_ingestion_artifact = DataIngestionArtifact
                 ) -> None:
        
        self.data_transform_config = data_transform_config
        self.data_ingestion_artifact = data_ingestion_artifact
        
        

    def create_Image_Generator_config(self,split='train'):
        if split=='train':
            train_datagen_params = {'config':
                        {'rescale':1./255,
                        'horizontal_flip':self.data_transform_config.horizontal_flip,
                        'rotation_range':self.data_transform_config.rotation_range,
                        'shear_range':self.data_transform_config.shear_range,
                        'zoom_range':self.data_transform_config.zoom_range}}
            return train_datagen_params['config']
        elif split == 'val':
            val_datagen_params = {'config':
                    {'rescale':1./255}}
            return val_datagen_params['config']
        elif split == 'test':
            test_datagen_params = {'config':
                    {'rescale':1./255}}
            return test_datagen_params['config']
        
    
    def initialize_config_save(self)->DataTransformArtifact:
        logging.info("Entered initialize_config_save method of DataTransform class")

        try:
            logging.info("Saving ImageDataGenerator API config for train,test and val sets")
            save_config(self.create_Image_Generator_config('train'),'train.json',self.data_transform_config.data_generator_dir)
            save_config(self.create_Image_Generator_config('val'),'val.json',self.data_transform_config.data_generator_dir)
            save_config(self.create_Image_Generator_config('test'),'test.json',self.data_transform_config.data_generator_dir)

            data_transform_artifact = DataTransformArtifact(test_data_generator_file_path = str(self.data_transform_config.data_generator_dir) + 'test.json',
                                                            val_data_generator_file_path= str(self.data_transform_config.data_generator_dir) + 'val.json',
                                                            train_data_generator_file_path= str(self.data_transform_config.data_generator_dir) + 'train.json')

            logging.info("Exited initialize_config_save method of DataTransform class")
            logging.info(f"Data Transform artifacts : {data_transform_artifact}")
            
            return data_transform_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def create_data_generator_config(self,split='train'):
        """Creating a function to generate Data_generator object config"""
        logging.info('Entered create_data_generator_config function of DataTransform')
        try:

            data_ingestion_file_path = self.data_ingestion_artifact.data_ingestion_file_path

            if split == 'train':
                data_dir = os.path.join(data_ingestion_file_path,'data','train')
            elif split == 'val':
                data_dir = os.path.join(data_ingestion_file_path,'data','train')
            elif split == 'test':
                data_dir = os.path.join(data_ingestion_file_path,'data','train')

            datagen_config = {'config' : {'directory':data_dir,
                                        'target_size':self.data_transform_config.target_size,
                                        'batch_size':self.data_transform_config.batch_size,
                                        'class_mode':self.data_transform_config.class_mode}}
            logging.info('Exited create_data_generator_config function of DataTransform')
            return datagen_config['config']
        
        except Exception as e:
            raise CustomException(e,sys)




    
    
    

                




