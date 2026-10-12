import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading

EAR_THRESHOLD = 0.21
CLOSED_TIME = 1.5
ALARM_INTERVAL = 2.0

mp_face_mesh = mp.solutions.face_mesh

left_eye = [33, 160, 158, 133, 153, 144]
right_eye = [362, 385, 387, 263, 373, 380]

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(points, indexes):
    p = [points[i] for i in indexes]
    vertical1 = distance(p[1], p[5])
    vertical2 = distance(p[2], p[4])
    horizontal = distance(p[0], p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal else 0

def play_alarm():
    try:
        winsound.Beep(1200, 350)
    except RuntimeError:
        winsound.MessageBeep()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

closed_start = None
last_alarm = 0
alert = False

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        h, w = frame.shape[:2]
        now = time.monotonic()

        status = "NORMAL"
        color = (0, 200, 0)

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            points = [(int(p.x * w), int(p.y * h))
                      for p in face.landmark]

            ear_left = eye_aspect_ratio(points, left_eye)
            ear_right = eye_aspect_ratio(points, right_eye)
            ear = (ear_left + ear_right) / 2

            for i in left_eye + right_eye:
                cv2.circle(frame, points[i], 2, (255, 255, 0), -1)

            if ear < EAR_THRESHOLD:
                if closed_start is None:
                    closed_start = now

                closed_duration = now - closed_start

                if closed_duration >= CLOSED_TIME:
                    status = "DROWSINESS ALERT!"
                    color = (0, 0, 255)
                    alert = True

                    if now - last_alarm >= ALARM_INTERVAL:
                        threading.Thread(
                            target=play_alarm,
                            daemon=True
                        ).start()
                        last_alarm = now
                else:
                    status = "EYES CLOSING"
                    color = (0, 165, 255)
                    alert = False
            else:
                closed_start = None
                alert = False

            cv2.putText(
                frame, f"EAR: {ear:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            if closed_start is not None:
                duration = now - closed_start
                cv2.putText(
                    frame, f"Eyes closed: {duration:.1f}s",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )
        else:
            closed_start = None
            alert = False
            status = "FACE NOT DETECTED"
            color = (0, 165, 255)

        cv2.putText(
            frame, status,
            (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )

        if alert:
            overlay = frame.copy()
            cv2.rectangle(
                overlay, (0, 0), (w, 110), (0, 0, 180), -1
            )
            frame = cv2.addWeighted(
                overlay, 0.35, frame, 0.65, 0
            )
            cv2.putText(
                frame, "WAKE UP! STOP SAFELY!",
                (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 3
            )

        cv2.imshow("DrowsyGuard - Driver Monitoring", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            closed_start = None
            last_alarm = 0
            alert = False

cap.release()
cv2.destroyAllWindows()
