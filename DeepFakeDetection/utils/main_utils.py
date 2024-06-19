import os,sys
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from DeepFakeDetection.entity.config_entity import DataTransformConfig
from DeepFakeDetection.entity.artifacts_entity import DataIngestionArtifact

from DeepFakeDetection.exception import CustomException
from DeepFakeDetection.logger import logging

# data_ingestion_artifact = DataIngestionArtifact()


def save_config(config, filename, directory):
        """
        Save a configuration dictionary as a JSON file.

        Args:
            config (dict): The configuration dictionary to be saved.
            filename (str): The name of the file to save the configuration.
            directory (str): The directory path where the file should be saved.
        """
        config_path = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f)



def load_config(filename, directory):
    """
    Load a configuration dictionary from a JSON file.

    Args:
        filename (str): The name of the file containing the configuration.
        directory (str): The directory path where the file is located.

    Returns:
        dict: The loaded configuration dictionary.
    """
    config_path = os.path.join(directory, filename)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# def create_data_generator_config(split='train'):
#     try:
#         if split == 'train':
#             data_dir = os.path.join(data_ingestion_artifact.data_ingestion_file_path,'train')
#         elif split == 'val':
#             data_dir = os.path.join(data_ingestion_artifact.data_ingestion_file_path,'train')
#         elif split == 'test':
#             data_dir = os.path.join(data_ingestion_artifact.data_ingestion_file_path,'train')

#         datagen_config = {'config' : data_dir,
#                                     'target_size':data_transform_config.target_size,
#                                     'batch_size':data_transform_config.batch_size,
#                                     'class_mode':data_transform_config.class_mode}
        
#         return datagen_config['config']
    
#     except Exception as e:
#         raise CustomException(e,sys)
    