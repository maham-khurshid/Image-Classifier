import streamlit as st
import numpy as np
import cv2
import joblib
import json
import tempfile
import pywt
import os

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
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3,1), scalled_img_har.reshape(32 * 32,1)))
        return combined_img.reshape(1, -1)
    else:
        return None

# Create the Streamlit app
st.title("Celebrity Image Classifier")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        image = preprocess_image(uploaded_image)

        if image is not None:
            prediction = best_clf.predict(image)
            predicted_class = list(class_dict.keys())[list(class_dict.values()).index(prediction[0])]

            st.write(f"Predicted Celebrity: {predicted_class.split('/')[-1]}")

            prediction_proba = best_clf.predict_proba(image)
            st.write(f"Prediction Score: {np.max(prediction_proba)}")