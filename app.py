import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

# Set environment variables to reduce TensorFlow verbosity
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING messages
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations warnings

# # Suppress TensorFlow warnings
# import warnings

# warnings.filterwarnings("ignore")
# tf.get_logger().setLevel("ERROR")

# Streamlit app code - MUST BE FIRST
st.set_page_config(
    page_title="Pneumonia Prediction from Chest X-Ray images",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Subhranil2004/pneumonia-classification#live-demo",
        "Report a bug": "https://github.com/Subhranil2004/pneumonia-classification/issues",
    },
)


# Load the saved model
@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading AI model..."):
            model = tf.keras.models.load_model(
                r"./Model/vgg_model.keras", compile=False
            )
        # st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


model = load_model()


def preprocess_image(image_path):
    """Preprocess uploaded image for model prediction"""
    # Load the image with the target size
    img = image.load_img(image_path, target_size=(224, 224))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image using VGG16 preprocess_input function
    preprocessed_img = preprocess_input(img_array)

    return preprocessed_img


# Define a function for model inference
def predict(image):
    if model is None:
        st.error(
            "❌ Model not loaded. Please check if the model file exists and reload the page."
        )
        return None

    try:
        with st.spinner("Analyzing X-ray image..."):
            # Open and preprocess the image
            preprocessed_image = preprocess_image(image)

            # Convert image to NumPy array
            img_array = np.array(preprocessed_image)

            # Make predictions using the loaded model
            prediction = model.predict(
                img_array, verbose=0
            )  # verbose=0 to suppress output

        return prediction
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None


# Sidebar
with st.sidebar:
    st.image(
        "./Images/image.jpg",
        width="stretch",
        # use_container_width=True, # deprecated ^1.49.1
        output_format="JPEG",
    )

st.sidebar.title("🩺 Pneumonia Detection")
st.sidebar.write(
    "This AI model is trained on the ***Pneumonia X-Ray Images dataset*** from [**Kaggle**](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images) and uses a Convolutional Neural Network with data augmentation."
)

st.sidebar.link_button("GitHub", "https://github.com/Subhranil2004")

st.markdown(
    f"""
        <style>
            .sidebar {{
                width: 500px;
            }}
        </style>
    """,
    unsafe_allow_html=True,
)

# Main content
st.title("🩺 Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray image to get an AI-powered diagnosis")

uploaded_file = st.file_uploader(
    "📁 Choose a chest X-Ray image...",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Supported formats: JPG, PNG, BMP, TIFF",
)

if uploaded_file is not None:
    # Display the uploaded image with border
    st.image(
        uploaded_file,
        caption="📸 Uploaded X-Ray Image",
        width=300,
        clamp=True,
    )

    # Perform prediction
    if st.button("🔍 Predict", type="primary"):
        result = predict(uploaded_file)

        if result is not None:
            # Display the prediction result
            if result[0][0] > 0.5:
                output = ":red[PNEUMONIA DETECTED] ⚠️"
                conf = (result[0][0] - 0.5) * 2 * 100
            else:
                output = "NORMAL ✅"
                conf = (0.5 - result[0][0]) * 2 * 100

            st.success(f"**Prediction:** {output}")
            st.info(f"**Confidence:** {conf:.2f}%")
        else:
            st.error("Prediction failed. Please try again with a different image.")


expander = st.expander("📋 Sample X-ray images to try", expanded=True)
expander.write("👆 Just drag-and-drop your chosen image above")
sample_images = [
    "./Images/viral2.jpeg",
    "./Images/bacterial1.jpg",
    "./Images/viral1.jpg",
    "./Images/IM-0028-0001.jpeg",
    "./Images/person101_bacteria_484.jpeg",
    "./Images/person3_virus_17.jpeg",
]

cols = expander.columns(3)
for idx, img_path in enumerate(sample_images):
    with cols[idx % 3]:
        try:
            st.image(img_path, width=200)
        except Exception:
            st.error(f"Image not found: {img_path}")

expander = st.expander("📊 Training Results", expanded=False)
expander.image("./Images/confusion_matrix.png", caption="Confusion Matrix")
expander.image("./Images/report.png", caption="Classification Report")

# Footer
st.write("\n\n\n")
st.markdown("---")
st.markdown(
    "Drop in any discrepancies or give suggestions in `Report a bug` option within the `⋮` menu"
)

st.markdown(
    "<div style='text-align: right'> Developed with ❤️ by Subhranil Nandy </div>",
    unsafe_allow_html=True,
)
