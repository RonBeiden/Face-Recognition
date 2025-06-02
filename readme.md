# 🧠 Face Recognition Web App with Streamlit

This is a Streamlit-based web application for face recognition in **images** and **videos**, built using the powerful **InsightFace** library for face detection and feature embedding. The app identifies known individuals by comparing their facial embeddings to a pre-generated set.

---

## 📆 Key Features

* **📷 Image Upload**
  Upload a group photo, and the app will:

  * Detect all faces
  * Recognize known individuals
  * Draw labeled bounding boxes around them

* **🎞️ Video Upload**
  Upload a video file and the app will:

  * Extract all frames
  * Perform face recognition across frames
  * Return a final list of recognized individuals
    *(Note: Processed frames are not shown to maintain performance.)*

* **⚡ Performance Optimizations**

  * Frame skipping for faster video processing
  * GPU acceleration (`ctx_id=0` in InsightFace)
  * Fast embedding comparison using distance thresholds

---

## 🛠️ Installation & Usage

### 1. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Generate Known Face Embeddings

Before using the app, you must generate face embeddings for known individuals:

```bash
python train_faces.py
```

This will generate a `embeddings.pkl` file based on face images stored in the `train_data/` directory.

### 3. Launch the Web App

```bash
streamlit run app.py
```

---

## 💡 How to Use

### 🖼️ Tab 1: Image Upload

* Upload `.jpg`, `.jpeg`, or `.png` files
* Faces will be recognized and annotated in the displayed image

### 🎥 Tab 2: Video Upload

* Upload `.mp4`, `.avi`, or `.mov` files
* The app will analyze the video and return a list of all recognized individuals

---

## 🧠 Powered By

* **[InsightFace](https://github.com/deepinsight/insightface)** – state-of-the-art face detection & recognition
* **Streamlit** – rapid deployment of data apps

---

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit app
├── train_faces.py         # Script to generate face embeddings
├── embeddings.pkl         # Output of training script (face embeddings)
├── train_data/            # Images of known individuals (organized in folders)
│   ├── person1/
│   ├── person2/
│   └── person3/
└── requirements.txt       # Python dependencies
```

---

## 📃 License

For personal and educational use only. Some models used (e.g., InsightFace) may require separate licenses for commercial usage.
