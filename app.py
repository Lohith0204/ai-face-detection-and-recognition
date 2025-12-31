import streamlit as st
import cv2
import numpy as np
from ai_engine.detector import FaceDetector
from ai_engine.recognizer import FaceRecognizer

st.title("Face Detection & Recognition")

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

action = st.button("Detect & Recognize")

if action:

    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        image_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        faces = detector.detect_faces(image)
        names = recognizer.recognize(image, faces)
        for (x, y, w, h), name in zip(faces, names):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        st.subheader("Result")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )
