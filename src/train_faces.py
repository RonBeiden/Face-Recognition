# train_faces.py
import os
import numpy as np
import pickle
from insightface import app
from insightface.app import FaceAnalysis
import cv2

# Initialize InsightFace model for embedding extraction
face_app = FaceAnalysis(name='buffalo_l')  # SOTA face recognition model
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = face_app.get(img)
    if faces:
        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)  # ✅ Normalize embedding
        return emb    
    else:
        print(f"No face detected in {image_path}")
        return None

def main(train_dir='train_data', save_path='embeddings.pkl'):
    embeddings = {}

    for person_name in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_embeddings = []
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            emb = get_embedding(img_path)
            if emb is not None:
                person_embeddings.append(emb)

        if person_embeddings:
            # ✅ Average embeddings and store single vector per person
            embeddings[person_name] = np.mean(person_embeddings, axis=0)
            print(f"Processed {person_name} with {len(person_embeddings)} embeddings")
        else:
            print(f"Warning: No embeddings collected for {person_name}")

    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    main()
