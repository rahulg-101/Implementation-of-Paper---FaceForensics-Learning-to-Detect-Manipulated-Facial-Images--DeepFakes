from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_ingestion_file_path:str


@dataclass
class DataValidationArtifact:
    validation_status:str


@dataclass
class DataTransformArtifact:
    test_data_generator_file_path:str
    val_data_generator_file_path:str
    train_data_generator_file_path:str