# FastAPI Backend for Fake/Real Image & Video Detection

This is a FastAPI backend using pre-trained ML models (Xception, YOLOv8, LSTM) to detect whether an uploaded image or video is real or fake.

## 🔧 Features
- Image & video upload
- YOLOv8-based human detection
- Xception classification
- LSTM for video classification
- CORS enabled
- Contact form logging

## 📦 Requirements
- Railway (or Render, etc.)
- Python 3.10+

## 🚀 Deploy on Railway

1. Fork or clone this repo
2. Create a new Railway project
3. Add environment variable:
   - `RAILWAY_ENVIRONMENT` or your own like `MODEL_DOWNLOAD=true`
4. Deploy

## 📂 Model IDs (Google Drive)
- Human With Xception: `1wblGsFG8Y2zqhuLnTVK8ps73Uc55yP80`
- Non-Human With Xception: `17V_RuN6skVc3n9MH7hGkGeO_82k7ECxF`
- Video With Xception & LSTM: `1vT-EZAjA-6GnjMfiYvP9-pmHhBm_Aw8Q`
- YOLOv8n: `1sI1LWFDJRwJqaN4HO5M4OYTRq_V-Dtl9`
