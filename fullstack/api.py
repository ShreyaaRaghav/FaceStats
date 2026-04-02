from fastapi import FastAPI, Request   # ✅ merged import (important)
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
from fer import FER
import time
import pandas as pd
from collections import defaultdict

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(os.listdir("./templates"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

templates = Jinja2Templates(directory="./templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html"
    )


@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request):
    return templates.TemplateResponse(
        request,
        "analytics.html"
    )

# SAME COLORS
EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "sad": (255, 0, 0),
    "neutral": (200, 200, 200),
    "happy": (0, 255, 0),
    "surprise": (0, 255, 255)
}

# ✅ FIX: Safe initialization
try:
    detector = FER(mtcnn=True)
except Exception as e:
    print("FER init failed:", e)
    detector = None

cap = None

session_data = []


def draw_emotion_bars(frame, emotions):
    h, w, _ = frame.shape

    panel_width = 260
    panel_x_start = w - panel_width - 10
    panel_y_start = 50

    bar_height = 14
    gap = 22
    max_bar_width = 140

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x_start - 10, panel_y_start - 30),
        (w - 5, panel_y_start + len(emotions) * gap + 10),
        (30, 30, 30),
        -1
    )

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "EMOTIONS",
                (panel_x_start, panel_y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

    for i, (emo, val) in enumerate(sorted_emotions):
        y = panel_y_start + i * gap
        color = EMOTION_COLORS.get(emo, (255, 255, 255))
        bar_len = int(val * max_bar_width)

        cv2.putText(frame, emo.capitalize(),
                    (panel_x_start, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)

        cv2.rectangle(frame,
                      (panel_x_start + 90, y),
                      (panel_x_start + 90 + max_bar_width, y + 14),
                      (60, 60, 60), -1)

        cv2.rectangle(frame,
                      (panel_x_start + 90, y),
                      (panel_x_start + 90 + bar_len, y + 14),
                      color, -1)

        cv2.putText(frame, f"{val:.2f}",
                    (panel_x_start + 90 + max_bar_width + 5, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 180, 180), 1)


def generate_frames():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)

    # ✅ FIX: camera check
    if not cap.isOpened():
        print("Camera not accessible")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # ✅ FIX: safe FER usage
        results = []
        if detector is not None:
            try:
                results = detector.detect_emotions(frame)
            except Exception as e:
                print("FER runtime error:", e)

        cv2.putText(frame, "Emotion Tracker",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        for face in results:
            emotions = face.get("emotions", {})
            if not emotions:
                continue

            dominant = max(emotions, key=emotions.get)
            confidence = emotions[dominant]

            session_data.append({
                "time": time.time(),
                "emotion": dominant,
                "confidence": confidence
            })

            (x, y, w, h) = face["box"]
            color = EMOTION_COLORS.get(dominant, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"{dominant} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            draw_emotion_bars(frame, emotions)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/data")
def get_data():
    return JSONResponse(session_data)


def build_transition_model(csv_path="session.csv"):
    # ✅ FIX: safe CSV loading
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)

    transitions = defaultdict(lambda: defaultdict(int))

    emotions = df["emotion"].tolist()

    for i in range(len(emotions) - 1):
        curr = emotions[i]
        nxt = emotions[i + 1]
        transitions[curr][nxt] += 1

    return transitions

@app.get("/predict")
def predict():
    if len(session_data) < 2:
        return {"current": "none", "predicted": "none"}

    transitions = defaultdict(lambda: defaultdict(int))

    emotions = [d["emotion"] for d in session_data]

    for i in range(len(emotions) - 1):
        transitions[emotions[i]][emotions[i+1]] += 1

    current = emotions[-1]

    if current not in transitions:
        return {"current": current, "predicted": "unknown"}

    predicted = max(transitions[current], key=transitions[current].get)

    return {
        "current": current,
        "predicted": predicted
    }