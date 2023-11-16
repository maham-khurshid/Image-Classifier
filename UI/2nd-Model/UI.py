import streamlit as st
import numpy as np
import cv2
import joblib
import json
import tempfile
import pywt
import os
from skimage.transform import resize

# Define a function for wavelet transformation
def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

# Load the saved model
best_clf = joblib.load('saved_model.pkl')

# Load the class dictionary
with open("class_dictionary.json", "r") as f:
    class_dict = json.load(f)

# Define a function to preprocess the image
def preprocess_image(uploaded_image):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(uploaded_image.read())

    img = cv2.imread(temp_filename)

    os.remove(temp_filename)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        img = resize(img, (32, 32), anti_aliasing=True)
        return img
    else:
        return None

# Create the Streamlit app
st.title("Celebrity Image Classifier")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Preprocess the uploaded image
    image = preprocess_image(uploaded_image)

    if image is not None:
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Uploaded Image:")

        # Make a prediction
        st.write("Predicting...")
        image = image.flatten()  # Flatten the image to a 1D array
        image = image.reshape(1, 4096)  # Reshape to match the model's input size (4096 features)
        prediction = best_clf.predict(image)
        prediction_score = best_clf.predict_proba(image).max()

        # Map the numeric label to celebrity name
        predicted_class = list(class_dict.keys())[list(class_dict.values()).index(int(prediction[0]) if prediction is not None else -1)]

        # Display the result
        st.write(f"Predicted Celebrity: {predicted_class}")
        st.write(f"Prediction Score: {prediction_score:.2f}")