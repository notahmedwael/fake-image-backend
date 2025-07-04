from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from PIL import Image
import io
import os
import datetime
import subprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

def download_model_gdown(gdrive_id: str, output_path: str):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Downloading {output_path}...")
        subprocess.run(["gdown", f"https://drive.google.com/uc?id={gdrive_id}", "-O", output_path])
        print(f"Downloaded {output_path}.")

if os.getenv("RAILWAY_ENVIRONMENT"):  # or use a custom one like "MODEL_DOWNLOAD"
    download_model_gdown("1wblGsFG8Y2zqhuLnTVK8ps73Uc55yP80", os.path.join(MODEL_DIR, "Human With Xception.keras"))
    download_model_gdown("17V_RuN6skVc3n9MH7hGkGeO_82k7ECxF", os.path.join(MODEL_DIR, "Non Human With Xception.keras"))
    download_model_gdown("1vT-EZAjA-6GnjMfiYvP9-pmHhBm_Aw8Q", os.path.join(MODEL_DIR, "Video With Xception & LSTM.keras"))
    download_model_gdown("1sI1LWFDJRwJqaN4HO5M4OYTRq_V-Dtl9", os.path.join(MODEL_DIR, "yolov8n.pt"))

# Load models
human_model = load_model(os.path.join(MODEL_DIR, "Human With Xception.keras"))
non_human_model = load_model(os.path.join(MODEL_DIR, "Non Human With Xception.keras"))
video_model = load_model(os.path.join(MODEL_DIR, "Video With Xception & LSTM.keras"))
yolo_model = YOLO(os.path.join(MODEL_DIR, "yolov8n.pt"))

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4"}

def preprocess_image(image: Image.Image, target_size: tuple):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def preprocess_video(video_path: str, target_size: tuple, num_frames: int = 10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret and len(frames) < num_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0
            frames.append(frame)

    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((target_size[0], target_size[1], 3)))
    return np.expand_dims(np.array(frames), axis=0)

@app.get("/")
async def root():
    return {"message": "FastAPI Server is running on Railway"}

@app.post("/predict/")
async def predict(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    predictions = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

        content = await file.read()

        if ext in ALLOWED_IMAGE_EXTENSIONS:
            image = Image.open(io.BytesIO(content)).convert("RGB")
            yolo_results = yolo_model(image)
            has_human = any(cls in [0] for cls in yolo_results[0].boxes.cls)

            model = human_model if has_human else non_human_model
            target_size = (224, 224) if has_human else (299, 299)
            input_data = preprocess_image(image, target_size)
        else:
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(content)
            input_data = preprocess_video(temp_path, (128, 128))
            os.remove(temp_path)
            model = video_model

        pred = model.predict(input_data)
        result = float(pred[0][0])
        label = "Real" if result > 0.5 else "Fake"

        predictions.append({
            "filename": file.filename,
            "label": label,
            "confidence": result
        })

    return predictions

@app.post("/contact/")
async def contact(name: str = Form(...), email: str = Form(...), message: str = Form(...)):
    if not all([name, email, message]) or '@' not in email:
        raise HTTPException(status_code=400, detail="Invalid input")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("contact_log.txt", "a") as f:
        f.write(f"{timestamp}|{name}|{email}|{message}\n")

    return {"message": "Contact form received"}
