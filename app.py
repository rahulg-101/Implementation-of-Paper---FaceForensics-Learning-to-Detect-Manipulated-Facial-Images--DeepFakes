import sys
import os
import gradio as gr
from PIL import Image
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN
import cv2

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

# Initialize MTCNN face detector
detector = MTCNN()

def extract_faces_from_video(video_path, target_size=(299, 299)):
    cap = cv2.VideoCapture(video_path)
    faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = detector.detect_faces(frame_rgb)

        for face in detected_faces:
            x, y, width, height = face['box']
            face_img = frame_rgb[y:y+height, x:x+width]
            face_img = cv2.resize(face_img, target_size)
            face_array = img_to_array(face_img) / 255.0
            faces.append(face_array)
    
    cap.release()
    return np.array(faces)

# Function to predict if the video is fake or real
def predict_video(video):
    faces = extract_faces_from_video(video)
    if len(faces) == 0:
        print("No faces detected in the video.")
        return None

    predictions = best_model.predict(faces)
    avg_prediction = np.mean(predictions)

    if avg_prediction > 0.5:
        return f"The video is predicted to be REAL with {avg_prediction} accuracy."
    else:
        return f"The video is predicted to be FAKE with {avg_prediction} accuracy."

    

title = "Detect Whether An Image is Real Or Fake"
description = """
The bot is trained on predicting whether a face image is a Fake one generated which can be misused or a Real one. Bring it on !!
"""

article = "Check out [the paper FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971v3) that this demo is 'subtly' based off of."

Images_pred = gr.Interface(
    fn=predict,
    inputs="image",
    outputs="text",
    title=title,
    description=description,
    article=article,
)

Video_pred = gr.Interface(predict_video,'video','text')

gr.TabbedInterface([Images_pred, Video_pred],['Images_Predictions', 'Video_pred']).launch()