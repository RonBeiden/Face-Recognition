import streamlit as st
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont
import cv2

# Load known embeddings
with open('embeddings.pkl', 'rb') as f:
    known_embeddings = pickle.load(f)

# Initialize model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

def classify_face(embedding, known_embeddings, threshold=0.9):
    embedding = embedding / np.linalg.norm(embedding)  # Normalize input embedding

    min_dist = float('inf')
    identity = "Unknown"
    for name, known_emb in known_embeddings.items():
        dist = np.linalg.norm(embedding - known_emb)
        print(f"Distance to {name}: {dist:.4f}")  # Debug output
        if dist < min_dist and dist < threshold:
            min_dist = dist
            identity = name
    return identity

st.title("Face Recognition on Group Photos")

uploaded_file = st.file_uploader("Upload a group photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_cv = np.array(img)[:, :, ::-1].copy()  # Convert to OpenCV BGR
    faces = face_app.get(img_cv)

    draw = ImageDraw.Draw(img)

    # Load font once (fallback if Arial not found)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        label = classify_face(embedding, known_embeddings)

        color = "green" if label != "Unknown" else "red"
        thickness = 6  # border thickness

        # Draw thicker rectangle by drawing multiple rectangles expanding outward
        for i in range(thickness):
            draw.rectangle(
                [(bbox[0]-i, bbox[1]-i), (bbox[2]+i, bbox[3]+i)],
                outline=color
            )

        # Calculate text size correctly with textbbox
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_position = (bbox[0], bbox[1] - text_height - 5)  # Slight padding above bbox

        # Draw filled rectangle behind text for readability
        draw.rectangle(
            [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
            fill=color
        )
        draw.text(text_position, label, fill="white", font=font)

    st.image(img, caption="Recognized faces")
else:
    st.write("Upload a photo with multiple people.")
