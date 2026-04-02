
#  EmotionSense AI  
### Real-Time Emotion Tracking + Predictive Intelligence

> “Not just detecting emotions — predicting what comes next.”

---

##  Overview

EmotionSense AI is a real-time emotion recognition system that uses your webcam to:

- Detect facial emotions live  
- Visualize emotional trends over time  
- Predict the *next emotion* using behavioral patterns  

This project combines **Computer Vision + Data Visualization + Predictive Modeling** into a seamless interactive experience.

---

## Features

### Live Emotion Detection
- Real-time face detection using FER  
- Emotion classification (7 emotions)  
- Visual overlays (bounding boxes + confidence)  

---

### Emotion Analytics Dashboard
- Interactive **line chart (Chart.js)**  
- Tracks emotion over time  
- Smooth timeline visualization  

---

### Emotion Prediction Engine
- Uses **Markov Chain logic**  
- Learns transitions between emotions  
- Predicts:  

```text
Current: Neutral → Next: Happy
````

* Updates live every 2 seconds ⚡

---

## Tech Stack

| Layer         | Tech Used                        |
| ------------- | -------------------------------- |
| Backend       | FastAPI                          |
| Frontend      | HTML, CSS, JavaScript            |
| Visualization | Chart.js                         |
| CV Model      | FER (Facial Emotion Recognition) |
| ML Logic      | Markov Chain (Custom)            |
| Data          | Pandas                           |

---

## Project Structure

```bash
FaceStats/
│
├── fullstack/
│   ├── api.py
│   ├── templates/
│   │   ├── index.html
│   │   └── analytics.html
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│
├── session.csv
├── requirements.txt
└── README.md
```

---

##  Installation

```bash
# Clone repository
git clone https://github.com/your-username/emotionsense-ai.git

# Navigate into backend
cd emotionsense-ai/backend

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run the App

```bash
python -m uvicorn api:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

##  How It Works

### 1. Emotion Detection

* Webcam frames → FER model
* Extracts:

  * dominant emotion
  * confidence score

---

### 2. Data Logging

Each frame stores:

```json
{
  "time": 1774631204.18,
  "emotion": "sad",
  "confidence": 0.62
}
```

---

### 3. Visualization

* Time-series line chart
* Emotion mapped to numeric index
* Smooth trend tracking

---

### 4. Prediction Logic (Core Innovation)

We use a **Markov Transition Model**:

```text
sad → neutral → neutral → happy
```

Transitions stored as:

```
P(next_emotion | current_emotion)
```

Prediction:

```python
predicted = max(next_emotions, key=next_emotions.get)
```

---

##  Why This Project is Unique

*  Real-time feedback loop *(detect → analyze → predict)*
*  Lightweight ML *(no heavy training required)*
*  Fast + explainable predictions
*  Combines CV + Analytics + ML in one system

---

##  Future Improvements

*  Deep Learning (LSTM for sequence prediction)
*  Mobile deployment
*  Cloud storage of sessions
*  Multi-person tracking
*  Emotion heatmaps

---

##  Demo Preview

* Live webcam feed + emotion overlays
* Real-time analytics graph
* Dynamic prediction updates

---

## 🛠️ Requirements

```txt
fastapi
uvicorn
opencv-python
fer
pandas
jinja2
```

---

##  Author

Built with chaos, caffeine, and questionable life choices 

---

##  If you like this project

Give it a ⭐ and flex it in your hackathon 

```
```
