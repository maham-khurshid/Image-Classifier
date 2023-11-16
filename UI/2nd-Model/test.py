import streamlit as st
import cv2
import numpy as np
import joblib
import json

# Load the trained model and class dictionary
model = joblib.load('saved_model.pkl')
with open("class_dictionary.json", "r") as f:
    class_dict = json.loads(f.read())

# Define the face_cascade and other required functions here
face_cascade = cv2.CascadeClassifier(".\\opencv\\haarcascade\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(".\\opencv\\haarcascade\\haarcascade_eye.xml")

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            return roi_color
    return None

def classify_image(image):
    scalled_raw_img = cv2.resize(image, (32, 32))
    img_har = w2d(image, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
    
    result = model.predict(combined_img.T)
    prob = model.predict_proba(combined_img.T)
    max_prob = np.max(prob)
    
    if max_prob < 0.5:
        return "No celebrity recognized", 0.0
    
    celeb_name = [k for k, v in class_dict.items() if v == result][0]
    return celeb_name, max_prob

# Streamlit UI
st.title("Celebrity Image Classification")
st.sidebar.title("Upload Image")

uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        celeb_name, prob = classify_image(image)
        st.write(f"The celebrity is {celeb_name} with a probability of {prob:.2f}")