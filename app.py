import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

st.title("MNIST Digit Classifier")
st.write("Upload a handwritten digit image (28x28 px, grayscale)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess the image for prediction
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.write(f"### 🧾 Predicted Digit: **{predicted_label}**")
