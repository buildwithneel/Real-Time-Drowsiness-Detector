import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
from collections import deque

CLOSE_THRESHOLD = 0.20
OPEN_THRESHOLD = 0.24
CLOSED_TIME = 1.5
ALARM_INTERVAL = 2.0
SMOOTH_FRAMES = 5

mp_face_mesh = mp.solutions.face_mesh

left_eye = [33, 160, 158, 133, 153, 144]
right_eye = [362, 385, 387, 263, 373, 380]

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(points, indexes):
    p = [points[i] for i in indexes]
    v1 = distance(p[1], p[5])
    v2 = distance(p[2], p[4])
    h = distance(p[0], p[3])
    return (v1 + v2) / (2 * h) if h else 0

def play_alarm():
    try:
        winsound.Beep(1200, 350)
    except RuntimeError:
        winsound.MessageBeep()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

ear_values = deque(maxlen=SMOOTH_FRAMES)
closed_start = None
last_alarm = 0
eyes_closed = False
alarm_running = False

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, frame = cap.read()

        if not success:
            print("Error: Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        now = time.monotonic()

        status = "NORMAL"
        color = (0, 200, 0)
        ear = None
        alert = False

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            points = [
                (int(p.x * w), int(p.y * h))
                for p in face.landmark
            ]

            ear_left = eye_aspect_ratio(points, left_eye)
            ear_right = eye_aspect_ratio(points, right_eye)
            raw_ear = (ear_left + ear_right) / 2

            ear_values.append(raw_ear)
            ear = sum(ear_values) / len(ear_values)

            for i in left_eye + right_eye:
                cv2.circle(frame, points[i], 2, (255, 255, 0), -1)

            if not eyes_closed and ear < CLOSE_THRESHOLD:
                eyes_closed = True
                closed_start = now

            elif eyes_closed and ear > OPEN_THRESHOLD:
                eyes_closed = False
                closed_start = None

            if eyes_closed and closed_start is not None:
                duration = now - closed_start

                if duration >= CLOSED_TIME:
                    status = "DROWSINESS ALERT!"
                    color = (0, 0, 255)
                    alert = True

                    if now - last_alarm >= ALARM_INTERVAL:
                        if not alarm_running:
                            alarm_running = True

                            def alarm_task():
                                global alarm_running
                                try:
                                    play_alarm()
                                finally:
                                    alarm_running = False

                            threading.Thread(
                                target=alarm_task,
                                daemon=True
                            ).start()

                        last_alarm = now

                else:
                    status = "EYES CLOSING"
                    color = (0, 165, 255)

                cv2.putText(
                    frame,
                    f"Closed: {duration:.1f}s",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            cv2.putText(
                frame,
                "Close: 0.20 | Open: 0.24",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

        else:
            ear_values.clear()
            eyes_closed = False
            closed_start = None
            status = "FACE NOT DETECTED"
            color = (0, 165, 255)

        cv2.putText(
            frame,
            status,
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )

        if alert:
            cv2.rectangle(frame, (0, 0), (w, 135), (0, 0, 180), 2)
            cv2.putText(
                frame,
                "WARNING: EYES CLOSED TOO LONG",
                (20, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        cv2.imshow("DrowsyGuard - Phase 2", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            ear_values.clear()
            closed_start = None
            eyes_closed = False
            last_alarm = 0

cap.release()
cv2.destroyAllWindows()
