from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_ingestion_file_path:str


@dataclass
class DataValidationArtifact:
    validation_status:str
