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

