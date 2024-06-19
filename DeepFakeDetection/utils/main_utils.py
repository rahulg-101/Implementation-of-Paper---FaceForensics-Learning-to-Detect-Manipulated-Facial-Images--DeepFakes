import os,sys
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model, load_model   # type: ignore
from tensorflow.keras.layers import SeparableConv2D,BatchNormalization,Conv2D,ReLU,MaxPooling2D,GlobalAveragePooling2D,Dense,Add,Dropout    # type: ignore
from tensorflow.keras.layers import RandomFlip, RandomRotation,Rescaling,Reshape    # type: ignore
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



"""
Constructing a base model using the Xception CNN model architecture
"""


def data_augmentor():
    """Augment the data by applying Flipping and Rotations to the Image"""
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05)
    ], name="data_augmentation")
    return data_aug

def Block(input_tensor, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
    x = input_tensor
    if out_filters != in_filters or strides != 1:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False)(x)
        residual = BatchNormalization()(residual)
    else:
        residual = x
    
    if grow_first:
        x = ReLU()(x)
        x = SeparableConv2D(out_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        in_filters = out_filters
    
    for _ in range(reps - 1):
        x = ReLU()(x)
        x = SeparableConv2D(in_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
    
    if not grow_first:
        x = ReLU()(x)
        x = SeparableConv2D(out_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
    
    if strides != 1:
        x = MaxPooling2D(3, strides=strides, padding='same')(x)
    
    x = Add()([x, residual])
    return x

data_augmentor = data_augmentor()

def Xception_Constructed(input_shape,data_augmentor = data_augmentor):
    ''' Define a tf.keras model for binary classification using Xception model
    Arguments:
        image_shape -- Image width,height and channels
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    '''
    inputs = tf.keras.Input(shape=input_shape)
        
    # apply data augmentation to the inputs
    x = data_augmentor(inputs)
    print(x.shape)
    
#     x = Reshape((x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
    
    # Entry Flow
    x = Conv2D(filters=32,kernel_size=(3,3),strides = 2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(64,(3,3),padding = 'valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Block(x, 64, 128, 2, strides=2, start_with_relu=False, grow_first=True)
    x = Block(x, 128, 256, 2, strides=2, start_with_relu=True, grow_first=True)
    x = Block(x, 256, 728, 2, strides=2, start_with_relu=True, grow_first=True)
    
    
    # Middle Flow
    for _ in range(8):
        x = Block(x, 728, 728, 3, start_with_relu=True, grow_first=True)
    
    # Exit Flow
        
    x = Block(x, 728, 1024, 3,strides=2,start_with_relu=True, grow_first=False)
    
    x = SeparableConv2D(1536,(3,3))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = SeparableConv2D(2048,(3,3))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    output_tensor = Dense(1,activation='sigmoid')(x)
    
    
    model = Model(inputs=inputs, outputs=output_tensor)
    return model 


