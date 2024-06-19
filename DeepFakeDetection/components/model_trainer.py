
import tensorflow as tf
from tensorflow.keras.models import Model, load_model   # type: ignore
from tensorflow.keras.optimizers import Adam    # type: ignore
from tensorflow.keras.models import Model, load_model   # type: ignore
from tensorflow.keras.layers import SeparableConv2D,BatchNormalization,Conv2D,ReLU,MaxPooling2D,GlobalAveragePooling2D,Dense,Add,Dropout    # type: ignore
from tensorflow.keras.layers import RandomFlip, RandomRotation,Rescaling,Reshape    # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers.schedules import PolynomialDecay # type: ignore
from tensorflow.keras.applications import Xception,xception # type: ignore
from dataclasses import dataclass

import pandas as pd

import os,sys

from DeepFakeDetection.exception import CustomException
from DeepFakeDetection.logger import logging

from DeepFakeDetection.entity.config_entity import ModelTrainerConfig
from DeepFakeDetection.entity.artifacts_entity import ModelTrainerArtifacts
from DeepFakeDetection.utils.main_utils import Xception_Constructed



class ModelTrainer:
    def __init__(self, model_trainer_config=ModelTrainerConfig, generator_config=None):
        self.model_trainer_config = model_trainer_config
        self.generator_config = generator_config

    
    def data_augmentor(self):
    
        """
        Augment the data by applying Flipping and Rotations to the Image
        """

        data_aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05)
        ], name="data_augmentation")
        return data_aug

    def learning_rate_scheduler(self):
        starter_learning_rate = 0.1
        end_learning_rate = 5e-5
        decay_steps = 10000
        learning_rate_fn = PolynomialDecay(starter_learning_rate,
                                           decay_steps,
                                           end_learning_rate,
                                           power=0.5)
        
        return learning_rate_fn
    
    def config(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_scheduler())
        
        config={
            "optimizer": optimizer,  # Use the string identifier for the optimizer
            "loss": "binary_crossentropy",
            "metric": "accuracy",
            "epoch": 1,
            "batch_size": 32,
        }
        return config
    


    def initiate_model_trainer(self)->ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            train_datagen = ImageDataGenerator(**self.generator_config['train_datagen_params'])
            val_datagen = ImageDataGenerator(**self.generator_config['val_datagen_params'])
            test_datagen = ImageDataGenerator(**self.generator_config['test_datagen_params'])

            train_generator = train_datagen.flow_from_directory(**self.generator_config['train'])
            val_generator = val_datagen.flow_from_directory(**self.generator_config['val'])
            test_generator = test_datagen.flow_from_directory(**self.generator_config['test'])    

            df = pd.DataFrame(columns=['Model', 'Accuracy'])
            logging.info("Training Baseline model")

            data_augmentor = self.data_augmentor()
            baseline_model = Xception_Constructed(input_shape=(299,299,3), data_augmentor=data_augmentor)

            baseline_model.compile(loss=self.config()['loss'],optimizer = self.config()['optimizer'],metrics=[self.config()['metric']])
            
            history = baseline_model.fit(train_generator,
                                        epochs=self.config()['epoch'],
                                        batch_size=self.config()['batch_size'],
                                        validation_data=val_generator)

            loss,accuracy = baseline_model.evaluate(test_generator)
            logging.info(f"Trained Baseline model and achieved {loss} loss and {accuracy} accuracy")
            df = df.append({'Model': baseline_model, 'Accuracy': accuracy}, ignore_index=True)


            logging.info(f"Fintuning pretrained Xception model")
            base_model = Xception(input_shape=(299,299,3),include_top=False,weights='imagenet',pooling ='avg')

            base_model.trainable = True

            # Fine-tune from this layer onwards
            fine_tune_at = 110

            # Freeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
                
            x = base_model.output

            predicted = Dense(1, activation="sigmoid")(x) 

            model = Model(base_model.input,predicted)
                
            model.compile(loss=self.config()['loss'],
                        optimizer = self.config()['optimizer'],
                        metrics=[self.config()['metric']])

            history = model.fit(
                train_generator,
                epochs=self.config()['epoch'],
                batch_size=self.config()['batch_size'],
                validation_data=val_generator,
                
            )

            loss,accuracy = model.evaluate(test_generator)
            logging.info(f"Finetuned pretrained model and achieved {loss} loss and {accuracy} accuracy")
            new_row = {'Model':Model,"Accuracy":accuracy}
            df = df.append({'Model': model, 'Accuracy': accuracy}, ignore_index=True)

            best_model_series = df.sort_values(by=['Accuracy'], ascending=False)['Model']
            best_model = best_model_series.iloc[0]
            
            os.makedirs(self.model_trainer_config.model_trainer_dir,exist_ok=True)
            

            best_model.save(os.path.join(self.model_trainer_config.model_trainer_dir,f'{self.model_trainer_config.best_pretrained_model}'))

            A = os.path.join(self.model_trainer_config.model_trainer_dir,self.model_trainer_config.best_pretrained_model)
            model_trainer_artifact = ModelTrainerArtifacts(trained_model_file_path=A)

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise CustomException(e, sys)








                    