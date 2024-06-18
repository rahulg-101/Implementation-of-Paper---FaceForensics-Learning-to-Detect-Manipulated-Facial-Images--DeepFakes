import sys,os

from DeepFakeDetection.pipeline.training_pipeline import TrainingPipeline


def trainRoute():
    obj = TrainingPipeline()
    obj.run_pipeline()
    return "Training Successfull!!" 

trainRoute()