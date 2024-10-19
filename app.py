import streamlit as st
import joblib
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Banana Leaf Disease Detection", page_icon="🍌")

def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to match the input shape of the model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array

# Load the trained model
model = joblib.load('cnn_model-1.sav')

# Define the class labels
class_labels = ['Cordana', 'Pestalotiopsis', 'Healthy', 'Sigatoka']

# Custom CSS for sidebar and page styling
st.markdown(
    """
    <style>
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #f4e04d;  /* Banana yellow background */
        color: #4b3e1e;  /* Dark brown text for contrast */
        font-family: 'Arial', sans-serif;
        padding: 20px;  /* Add padding for better alignment */
    }

    /* Ensure all sidebar text is dark brown */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div {
        color: #4b3e1e;
    }

    /* Center the sidebar image */
    .sidebar-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 70%;  /* Adjust the size of the image */
        margin-bottom: 20px;  /* Add some space below the image */
        border-radius: 50%; /* Make the image circular */
    }

    /* Style for buttons */
    .stButton > button {
        color: #f4e04d;  /* Banana yellow text */
        background-color: #4b3e1e;  /* Dark brown button background */
        border: 2px solid #f4e04d;  /* Banana yellow border */
        font-weight: bold;
        border-radius: 10px; /* Rounded corners */
    }

    /* Page title styling */
    .stApp h1 {
        color: #4b3e1e;  /* Dark brown title color */
        font-family: 'Georgia', serif;
        font-size: 2.5em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar content with optional logo/image
st.sidebar.title("🍌 Banana Leaf Disease Classification")
st.sidebar.write("**Model Capabilities:**")
st.sidebar.write("- Detects: **Cordana**, **Healthy**, **Pestalotiopsis**, **Sigatoka**")
st.sidebar.write("- Easy and fast disease detection for better crop management.")

# Streamlit app title
st.title("Welcome to Banana Leaf Disease Detection 🍌")
st.write(
    "Upload an image of a banana leaf, and our model will predict the likelihood of the leaf being affected by one of the following diseases: **Cordana**, **Pestalotiopsis**, **Sigatoka**, or classify it as **Healthy**."
)

# File uploader
uploaded_file = st.file_uploader("Upload a banana leaf image", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    if st.button("Predict"):
        progress_text = "Analyzing the image Please wait....."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        # Display the prediction
        st.write(f"Prediction: **{class_labels[predicted_class]}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.success(f'Hey! The model predicted **{class_labels[predicted_class]}**!', icon="✅")
