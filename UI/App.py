import streamlit as st
import numpy as np
import base64
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd

# Load your trained model
model = keras.models.load_model('model.keras')

# Define the Streamlit app
st.set_page_config(
    page_title='Celebrity Image Classifier',
    page_icon='📷'
)

# Sidebar
st.sidebar.header('Choose any of the following celeb Image to upload')
st.sidebar.image(".\\new.jpg", use_column_width=True)

# Homepage
st.header("Celebrity Image Classifier")
#st.write("Upload an image to classify the celebrity.")

uploaded_image = st.file_uploader("Upload an image to classify the celebrity", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Read and preprocess the uploaded image
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make a prediction
    prediction = model.predict(img)
    celebrity_classes = ['dwaynejohnson', 'emmawatson', 'johnnydepp', 'ladygaga', 'leonardodicaprio']
    predicted_class = celebrity_classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Display the result
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image(uploaded_image, caption="Uploaded Image", width=160)
    st.write(
        f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center;">'
        f'<h3>Predicted Celebrity: {predicted_class}</h3>'
        f'<p>Confidence: {confidence:.2%}</p>'
        f'</div>',
        unsafe_allow_html=True
    )