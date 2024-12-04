import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('pose_detection_model.h5')

model = load_model()

CLASS_NAMES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

def preprocess_image(image):
    image = image.convert("RGB") 
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  # adding batch dimension
    return image


def provide_feedback(predicted_class, confidence):
    feedback = {
        "downdog": "Ensure your spine is straight, hands firmly grounded, and heels pointing downward.",
        "goddess": "Check your knee alignment—knees should point outwards. Engage your core.",
        "plank": "Maintain a straight line from head to heels. Avoid sagging hips.",
        "tree": "Focus on balance—keep your core engaged and your standing leg straight.",
        "warrior2": "Ensure your front knee is aligned with your toes and your arms are parallel to the floor."
    }
    return f"{predicted_class} pose detected with {confidence:.2f}% confidence.\n\nCorrection Tip: {feedback.get(predicted_class, 'No specific feedback available.')}"
    
    
    
st.title("Yoga Pose Detection & Correction")
st.write("Upload an image of yourself performing a yoga pose to get feedback on alignment and accuracy.")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing the pose..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        confidence = np.max(predictions) * 100
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.success("Pose Detection Complete!")
    st.write(provide_feedback(predicted_class, confidence))
