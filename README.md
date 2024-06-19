# Implementation-of-Paper---FaceForensics-Learning-to-Detect-Manipulated-Facial-Images--DeepFakes

This project is an implementation of the paper [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971v3) for detecting fake or manipulated facial images. The project follows a modular and production-grade approach, divided into various components such as config files, data ingestion, data validation, data transformation, model training, and a training pipeline.

## Use Case

With the advancement of deep learning techniques, it has become increasingly easy to create fake or manipulated images, which can be used for malicious purposes such as spreading misinformation or identity theft. This project aims to develop a model that can accurately detect whether a given facial image is real or fake, providing a confidence score for the prediction.

## Project Structure

The project is organized into the following components:

1. **Config Files**: Contains configuration settings for various components of the project.
2. **Data Ingestion**: Handles the process of downloading and ingesting the dataset.
3. **Data Validation**: Validates the ingested data to ensure its integrity.
4. **Data Transformation**: Applies necessary transformations and augmentations to the dataset for model training.
5. **Model Trainer**: Implements two approaches for training the model:
    - Building a model from scratch and training it from the beginning.
    - Using a pre-trained Xception model and fine-tuning it for the task.
6. **Training Pipeline**: Orchestrates the end-to-end training process by integrating all the components.
7. **Logger and Custom Exception**: Implements logging and custom exception handling mechanisms.
8. **Experimentation Notebook**: Contains serialized code for experimenting with the project components.
9. **Web Application**: Provides a user-friendly interface built with Gradio for uploading images and obtaining predictions.

## Prerequisites

- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)


## Installation

1. Clone the repository:

`git clone https://github.com/rahulg-101/Implementation-of-Paper---FaceForensics-Learning-to-Detect-Manipulated-Facial-Images--DeepFakes`

2. Navigate to the project directory:

`cd deep-fake-image-detection`

3. Install the required Python packages:

`pip install -r requirements.txt`


## Usage

1. Ensure that the dataset is downloaded and properly ingested into the project.
2. Run the training pipeline to train the model:

```python
python training_pipeline.py
```

3. After successful training, run the web application:

```python
python app.py
```

4. The web application will provide a user interface for uploading images and obtaining predictions.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project is based on the research paper [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971v3) by Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner.

## Curious Cases

If you're interested in creating projects that teach you how to develop production-grade code, then you should check out my previous project, "[Objection Detection and Tracking on Football Players](https://github.com/rahulg-101/Objection-Detection-and-Tracking-on-Football-Players) ," where I have outlined all the necessary steps to build such a project.

#### Thank You
