import sys,os

from DeepFakeDetection.pipeline.training_pipeline import TrainingPipeline


def trainRoute():
    obj = TrainingPipeline()
    train_generator,val_generator,test_generator = obj.run_pipeline()
    image_batch, label_batch = next(iter(train_generator))
    print(image_batch.shape)
    return "Training Successfull!!" 

trainRoute()

