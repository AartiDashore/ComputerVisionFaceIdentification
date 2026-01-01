import cv2
from deepface import DeepFace

# -----------------------------
# Configuration
# -----------------------------
ANALYZE_EVERY_N_FRAMES = 30   # increase for more FPS (try 10–15)
FONT = cv2.FONT_HERSHEY_SIMPLEX

WINDOW_NAME = "AI Face Analysis System"

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)
frame_count = 0

# Cached results (for FPS optimization)
cached_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ----------------------------------
    # Run DeepFace every N frames only
    # ----------------------------------
    if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
        try:
            cached_results = DeepFace.analyze(
                img_path=rgb_frame,
                actions=['age', 'gender', 'emotion', 'race'],
                enforce_detection=False,
                detector_backend='opencv'
            )
        except Exception:
            cached_results = []

    # ----------------------------------
    # Draw results
    # ----------------------------------
    if cached_results:
        for result in cached_results:
            region = result.get("region", {})

            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            age = result.get("age", "N/A")
            gender = result.get("dominant_gender", "N/A")
            emotion = result.get("dominant_emotion", "N/A")
            ethnicity = result.get("dominant_race", "N/A")

            # ---- Face shape (simple geometric approximation) ----
            face_shape = "Oval" if h > w else "Round"

            # ---- Draw bounding box ----
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ---- Display text ----
            y_offset = y - 70
            cv2.putText(frame, f"Age: {age}", (x, y_offset),
                        FONT, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Gender: {gender}", (x, y_offset + 15),
                        FONT, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y_offset + 30),
                        FONT, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Ethnicity: {ethnicity}", (x, y_offset + 45),
                        FONT, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Face Shape: {face_shape}", (x, y_offset + 60),
                        FONT, 0.5, (0, 255, 0), 1)

    # ----------------------------------
    # Show output
    # ----------------------------------
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Detect window close button
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
