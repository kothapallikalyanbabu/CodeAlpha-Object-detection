# VisionTrack AI — Real-Time Object Detection & Tracking System
### CodeAlpha Internship Task 4

A clean, modern, and intuitive computer vision web platform built in pure **HTML, CSS, and JavaScript**, accompanied by a **FastAPI** backend script.

---

## 🌟 Features

1. **Live Camera Object Detection**:
   - Access real-time webcam feed with live canvas bounding boxes and class labels.
   - Live FPS throughput counter and active detection counts.
   - Adjustable confidence threshold slider (20% - 95%).
   - One-click snapshot download with bounding box annotations.

2. **Image & Video File Detection**:
   - Upload any image or video file.
   - Automatic bounding box generation with confidence percentages.
   - Coordinates table detailing object classes, confidence scores, and box dimensions.

3. **History & Analytics**:
   - Telemetry statistics: total objects tracked, tracked classes, latency, and accuracy rate.
   - Session history log stored in browser `localStorage`.
   - Clear history button.

4. **Clean Corporate Design**:
   - General professional color palette (slate-50 background `#f8fafc`, crisp white `#ffffff` cards, royal blue `#2563eb`, dark slate text `#0f172a`).
   - 1-click Light / Dark theme toggle.

---

## 🚀 How to Run the Website

### Option 1: Direct in Browser (No Server Needed!)
Simply double-click **`index.html`** or right-click `index.html` -> **Open with Google Chrome** (or any browser). Everything runs immediately!

### Option 2: Using Any Lightweight Local Server (e.g., Python / Node / Live Server)
If you have VS Code "Live Server" extension, click **"Go Live"**.
Or run in terminal:
```bash
npx serve .
```

### Option 3: Backend API (Optional Python Server)
If you want to run the FastAPI detection server:
```bash
pip install -r requirements.txt
python app.py
```
API runs on `http://127.0.0.1:8000`.

---

## 📁 Project Structure
```
├── index.html        # Main HTML structure with 4 tabs
├── style.css         # Clean CSS design system & general colors
├── script.js         # Camera stream, canvas bounding boxes, & history logic
├── app.py            # FastAPI backend script
├── requirements.txt  # Python dependencies
└── README.md         # Documentation & submission guide
```
