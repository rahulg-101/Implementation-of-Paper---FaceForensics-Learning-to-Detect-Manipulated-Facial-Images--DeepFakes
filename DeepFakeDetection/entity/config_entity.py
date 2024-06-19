import os
from dataclasses import dataclass
from datetime import datetime
from DeepFakeDetection.constant.training_pipeline import *

@dataclass
class TrainingPipelineConfig():
    artifacts_dir:str = ARTIFACTS_DIR

training_pipeline_config = TrainingPipelineConfig()     # This object stores the path to artifacts

@dataclass
class DataIngestionConfig:
    data_ingestion_dir = os.path.join(training_pipeline_config.artifacts_dir,DATA_INGESTION_DIR_NAME)

    dataset_name = DATASET_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir_name = os.path.join(training_pipeline_config.artifacts_dir,DATA_VALIDATION_DIR_NAME)

    valid_status_file_dir = os.path.join(data_validation_dir_name,DATA_VALIDATION_STATUS_FILE)

    required_file_list = DATA_VALIDATION_ALL_REQUIRED_FILES

@dataclass
class DataTransformConfig:
    data_generator_dir = os.path.join(training_pipeline_config.artifacts_dir,DATA_GENERATOR_DIR_NAME)

    horizontal_flip = HORIZONTAL_FLIP
    rotation_range = ROTATION_RANGE
    shear_range = shear_range
    zoom_range = ZOOM_RANGE



    target_size = TARGET_SIZE
    batch_size = BATCH_SIZE
    class_mode = CLASS_MODE

