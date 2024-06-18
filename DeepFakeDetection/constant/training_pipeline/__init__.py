ARTIFACTS_DIR = 'artifacts'

"""
Data Ingestion related constant, start with DATA_INGESTION Var Name,
this is useful when you want to change your directories or URL from which you are receiving the files.

"""

DATA_INGESTION_DIR_NAME = "data_ingestion"      # Store downloaded data in this directory

DATASET_NAME = 'farhansharukhhasan/faceforensics1600-videospreprocess'


"""
Data Validation & Transformation related constant, start with DATA_VALIDATION Var Name
"""

DATA_VALIDATION_DIR_NAME = "data_validation"    # Create data_validation folder

DATA_VALIDATION_STATUS_FILE = "status.txt"      # Return status as False or True

DATA_VALIDATION_ALL_REQUIRED_FILES = ["test",
                                    "train",
                                    "val"]

DATA_GENERATOR_DIR_NAME = "data_generator_dir"

TARGET_SIZE = (299,299)

BATCH_SIZE = 32

CLASS_MODE = 'binary'

