import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('flower_classifier_model.h5')

# Define class labels
class_labels = {0: 'daisy', 1: 'sunflower', 2: 'rose', 3: 'dandelion', 4: 'tulip'}


# Preprocessing function
def preprocess_image(uploaded_file):
    try:
        # Open the image file
        img = Image.open(uploaded_file).convert('RGB')

        # Resize to the target size
        img = img.resize((150, 150))

        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Expand dimensions to match model input
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None


# Streamlit interface
st.title("Flower Classification App")
st.write("Upload an image of a flower, and the model will classify it as Daisy, Sunflower, Rose, Dandelion, or Tulip.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(uploaded_file)

    if img_array is not None:
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Display results
        st.write(f"Prediction: *{class_labels.get(predicted_class, 'Unknown')}*")
        st.write(f"Confidence: *{confidence:.2f}%*")

        # Confidence threshold
        if confidence < 50:  # Adjust the threshold as needed
            st.write("The confidence is low; the result might not be accurate.")
