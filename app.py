import sys
import os
import gradio as gr
from PIL import Image
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model

# from DeepFakeDetection.pipeline.training_pipeline import TrainingPipeline

def load_best_model():
    # training_pipeline = TrainingPipeline()
    best_model_path = 'artifacts/model_trainer/best_pt.hdf5'
    best_model = load_model(best_model_path)
    return best_model

best_model = load_best_model()

def predict(im):
    # Load the image
    img = Image.fromarray(im.astype('uint8'), 'RGB')  # Convert to PIL Image

    # Preprocess the image
    img = img.resize((299, 299))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = best_model.predict(img_array)

    # Print prediction
    confidence_score = 1 - prediction[0][0]
    if prediction[0][0] < 0.5:
        result = f"Fake Image with a confidence score of {confidence_score * 100:.3f}%"
    else:
        result = f"Real Image with a confidence score of {confidence_score * 100:.3f}%"

    return result

title = "Detect Whether An Image is Real Or Fake"
description = """
The bot is trained on predicting whether a face image is a Fake one generated which can be misused or a Real one. Bring it on !!
"""

article = "Check out [the paper FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971v3) that this demo is 'subtly' based off of."

gr.Interface(
    fn=predict,
    inputs="image",
    outputs="text",
    title=title,
    description=description,
    article=article,
).launch()