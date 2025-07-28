import streamlit as st
import tensorflow as tf
import numpy as np
import rasterio
from PIL import Image

# Load model
model = tf.keras.models.load_model("forest_loss_updated_cnn.h5")

st.set_page_config(page_title="Forest Loss Detector", layout="centered")
st.title("🌲 Forest Loss Detection")

uploaded_file = st.file_uploader("Upload a multiband .tif image", type=["tif", "tiff"])

if uploaded_file is not None:
    with rasterio.open(uploaded_file) as src:
        img = src.read().astype(np.float32)  # shape: (bands, H, W)
        img = np.transpose(img, (1, 2, 0))   # shape: (H, W, bands)

    # Resize to match training input size (e.g., 128x128)
    img_resized = np.array(Image.fromarray(img.astype(np.uint8)).resize((128, 128)))

    if img_resized.shape[-1] != 4:
        st.error("Expected 4-band image (e.g., RGB + NIR).")
    else:
        # Normalize
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)  # (1, 128, 128, 4)

        # Predict
        prediction = model.predict(img_resized)[0][0]
        label = "🌳 No Loss" if prediction < 0.5 else "🔥 Forest Loss"
        st.success(f"**Prediction:** {label} ({prediction:.2f})")

        # Show image
        st.image(img_resized[0][:, :, :3], caption="Uploaded Image (RGB)", use_column_width=True)
