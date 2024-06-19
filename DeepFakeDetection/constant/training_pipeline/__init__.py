ARTIFACTS_DIR = 'artifacts'

"""
Data Ingestion related constant, start with DATA_INGESTION Var Name,
this is useful when you want to change your directories or URL from which you are receiving the files.

"""

DATA_INGESTION_DIR_NAME = "data_ingestion"      # Store downloaded data in this directory

DATASET_NAME = 'farhansharukhhasan/faceforensics1600-videospreprocess'


"""
Data Validation related constant, start with DATA_VALIDATION Var Name
"""

DATA_VALIDATION_DIR_NAME = "data_validation"    # Create data_validation folder

DATA_VALIDATION_STATUS_FILE = "status.txt"      # Return status as False or True

DATA_VALIDATION_ALL_REQUIRED_FILES = ["test",
                                    "train",
                                    "val"]

"""
Data Validation related constant, start with DATA_VALIDATION Var Name
"""

DATA_GENERATOR_DIR_NAME = "data_generator_dir"

HORIZONTAL_FLIP =  True
ROTATION_RANGE = 10
shear_range = 0.2
ZOOM_RANGE = 0.2


TARGET_SIZE = (299,299)

BATCH_SIZE = 32

CLASS_MODE = 'binary'


"""
Model trainer related consstant start with MODEL_TRAINER VAR Name
"""

MODEL_TRAINER_DIR_NAME = "model_trainer"    # Create model_trainer folder

MODEL_TRAINER_PRETRAINED_WEIGHT_NAME = "best_pt.hdf5"


