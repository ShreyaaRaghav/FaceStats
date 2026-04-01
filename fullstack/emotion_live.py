import cv2
from fer import FER
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "sad": (255, 0, 0),
    "neutral": (200, 200, 200),
    "happy": (0, 255, 0),
    "surprise": (0, 255, 255)
}

# Convert emotion to number 
EMOTION_INDEX = {k: i for i, k in enumerate(EMOTION_COLORS.keys())}


def analyze_session(data):
    if not data:
        print(" No faces detected during session.")
        return

    print("\n Session Summary")

    counts = Counter([row["emotion"] for row in data])
    total = len(data)

    for emotion, count in counts.items():
        print(f"{emotion:<10} → {(count / total) * 100:.2f}%")

    # Save CSV
    df = pd.DataFrame(data)
    df.to_csv("session.csv", index=False)
    print("session.csv saved!")

    # Plot timeline
    times = [row["time"] - data[0]["time"] for row in data]
    values = [EMOTION_INDEX[row["emotion"]] for row in data]

    plt.figure()
    plt.plot(times, values)
    plt.yticks(list(EMOTION_INDEX.values()), list(EMOTION_INDEX.keys()))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotion")
    plt.title("Emotion Timeline")
    plt.grid(True)
    plt.show()

def draw_emotion_bars(frame, emotions):
    """Clean right-side emotion panel"""
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

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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
                      (panel_x_start + 90 + max_bar_width, y + bar_height),
                      (60, 60, 60), -1)

        cv2.rectangle(frame,
                      (panel_x_start + 90, y),
                      (panel_x_start + 90 + bar_len, y + bar_height),
                      color, -1)

        cv2.putText(frame, f"{val:.2f}",
                    (panel_x_start + 90 + max_bar_width + 5, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 180, 180), 1)


def main():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    session_data = []

    print(" Starting Emotion Tracker (press 'q' to quit)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_emotions(frame)

        cv2.putText(frame, "LIVE EMOTION ANALYSIS",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        for face in results:
            emotions = face["emotions"]
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

        cv2.imshow("Emotion Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    analyze_session(session_data)


if __name__ == "__main__":
    main()