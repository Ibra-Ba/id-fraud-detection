import io

import streamlit as st
from PIL import Image

from src.api.predictor import predictor

# ── Setup ─────────────────────────────────────────
st.set_page_config(page_title="ID Fraud Detector", layout="centered")

st.title("🛂 ID Fraud Detection")
st.write("Upload an ID image to detect fraud.")

# ── Load model (once) ─────────────────────────────
if not predictor.is_loaded:
    with st.spinner("Loading model..."):
        predictor.load()

# ── Upload ────────────────────────────────────────
uploaded_file = st.file_uploader("Upload ID image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_bytes = uploaded_file.read()

    # Display image
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded ID", use_column_width=True)

    # ── Prediction ────────────────────────────────
    with st.spinner("Running inference..."):
        result = predictor.predict(image_bytes)

    # ── Results ───────────────────────────────────
    st.subheader("Prediction")

    label = result["label"]
    score = result["fraud_probability"]

    if label == "fraud":
        st.error(f"⚠️ Fraud detected (score={score:.4f})")
    else:
        st.success(f"✅ Genuine ID (score={score:.4f})")

    st.write(result)

    # ── Grad-CAM (only if fraud) ──────────────────
    if label == "fraud":
        st.subheader("Explainability (Grad-CAM)")

        with st.spinner("Generating explanation..."):
            try:
                cam = predictor.generate_gradcam(image_bytes)
                st.image(cam, caption="Fraud heatmap")
            except Exception as e:
                st.warning(f"Grad-CAM failed: {e}")
