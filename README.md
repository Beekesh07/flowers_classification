# Flower Classification App
### This is a web application built using Streamlit and TensorFlow that classifies flower images into one of five categories: Daisy, Sunflower, Rose, Dandelion, or Tulip. The model predicts the flower type based on the image uploaded by the user.

### Features
Upload an image of a flower.
The app uses a pre-trained TensorFlow model to classify the flower as one of the following:
Daisy
Sunflower
Rose
Dandelion
Tulip
The app displays the predicted flower type along with the confidence score.
If the model is not confident enough (below 50%), it will notify the user that the prediction was not found.
### Technologies Used
TensorFlow: For making model and making predictions.
Streamlit: For creating the user interface.
Pillow: For image processing.
NumPy: For numerical operations and array manipulation.

### Usage
Upload an image of a flower by clicking the "Choose an image..." button.
The app will classify the flower as either Daisy, Sunflower, Rose, Dandelion, or Tulip.
The app will display the predicted class label and the confidence score.
If the prediction confidence is below 50%, the app will notify you that the flower image doesn't belong to any of the trained classes.
### Model Details
The model used in this application is a simple CNN (Convolutional Neural Network) trained on a flower dataset. It accepts images of size 150x150 pixels and outputs one of the five classes.
