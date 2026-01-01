import cv2
from deepface import DeepFace
from collections import deque

cv2.namedWindow("AI Face Analysis System", cv2.WINDOW_NORMAL)

# -----------------------------
# Configuration
# -----------------------------
ANALYZE_EVERY_N_FRAMES = 30
SMOOTHING_WINDOW = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "AI Face Analysis System"

cap = cv2.VideoCapture(0)
frame_count = 0

age_history = deque(maxlen=SMOOTHING_WINDOW)
gender_history = deque(maxlen=SMOOTHING_WINDOW)
emotion_history = deque(maxlen=SMOOTHING_WINDOW)
race_history = deque(maxlen=SMOOTHING_WINDOW)

cached_results = []

print("Starting camera... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ----------------------------------
    # DeepFace analysis (reliable)
    # ----------------------------------
    if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
        try:
            cached_results = DeepFace.analyze(
                img_path=rgb_frame,
                actions=['age', 'gender', 'emotion', 'race'],
                detector_backend='mtcnn',   # ✅ reliable on CPU
                enforce_detection=False,    # ✅ no silent failure
            )

            r = cached_results[0]
            age_history.append(r["age"])
            gender_history.append(r["dominant_gender"])
            emotion_history.append(r["dominant_emotion"])
            race_history.append(r["dominant_race"])

        except Exception as e:
            print("DeepFace error:", str(e))
            cached_results = []

    if cached_results:
        result = cached_results[0]
        region = result.get("region", {})

        x, y, w, h = (
            region.get("x", 0),
            region.get("y", 0),
            region.get("w", 0),
            region.get("h", 0),
        )

        if w > 0 and h > 0:
            age = int(sum(age_history) / len(age_history))
            gender = max(set(gender_history), key=gender_history.count)
            emotion = max(set(emotion_history), key=emotion_history.count)
            ethnicity = max(set(race_history), key=race_history.count)

            face_shape = "Oval" if h > w else "Round"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            y0 = y - 80
            cv2.putText(frame, f"Age: {age}", (x, y0), FONT, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y0 + 20), FONT, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y0 + 40), FONT, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Ethnicity: {ethnicity}", (x, y0 + 60), FONT, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Face Shape: {face_shape}", (x, y0 + 80), FONT, 0.5, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
