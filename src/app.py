import streamlit as st
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import tempfile
from pathlib import Path

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
        if dist < min_dist and dist < threshold:
            min_dist = dist
            identity = name
    return identity

def extract_frames_from_video(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video.")
        return []

    frame_paths = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        count += 1
    cap.release()
    return frame_paths

def process_frame(frame_path):
    img_cv = cv2.imread(frame_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img_cv)

    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        label = classify_face(embedding, known_embeddings)
        color = "green" if label != "Unknown" else "red"
        thickness = 4

        for i in range(thickness):
            draw.rectangle(
                [(bbox[0]-i, bbox[1]-i), (bbox[2]+i, bbox[3]+i)],
                outline=color
            )

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (bbox[0], bbox[1] - text_height - 5)

        draw.rectangle(
            [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
            fill=color
        )
        draw.text(text_position, label, fill="white", font=font)

    return pil_img

# --- Streamlit UI ---
st.title("Face Recognition")

tab1, tab2 = st.tabs(["Image Upload", "Video Upload"])

# --- IMAGE TAB ---
with tab1:
    uploaded_file = st.file_uploader("Upload a group photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        img_cv = np.array(img)[:, :, ::-1].copy()
        faces = face_app.get(img_cv)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            label = classify_face(embedding, known_embeddings)
            color = "green" if label != "Unknown" else "red"
            thickness = 4
            for i in range(thickness):
                draw.rectangle([(bbox[0]-i, bbox[1]-i), (bbox[2]+i, bbox[3]+i)], outline=color)

            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_position = (bbox[0], bbox[1] - text_height - 5)
            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill=color
            )
            draw.text(text_position, label, fill="white", font=font)

        st.image(img, caption="Recognized faces")

# --- VIDEO TAB ---
with tab2:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            temp_video_path = tmp_file.name

        video_name = Path(uploaded_video.name).stem
        frame_folder = os.path.join(tempfile.gettempdir(), video_name)

        st.info("Extracting frames from video...")
        frame_paths = extract_frames_from_video(temp_video_path, frame_folder)

        if not frame_paths:
            st.error("No frames extracted from the video.")
        else:
            st.info("Running face recognition on selected frames...")
            recognized_names = set()

            for i, frame_path in enumerate(frame_paths[::2]):  # Every 10th frame
                img_cv = cv2.imread(frame_path)
                if img_cv is None:
                    continue
                faces = face_app.get(img_cv)
                for face in faces:
                    label = classify_face(face.embedding, known_embeddings)
                    if label != "Unknown":
                        recognized_names.add(label)

            if recognized_names:
                st.success("People recognized in the video:")
                for name in sorted(recognized_names):
                    st.markdown(f"- **{name}**")
            else:
                st.warning("No known individuals recognized in the video.")

