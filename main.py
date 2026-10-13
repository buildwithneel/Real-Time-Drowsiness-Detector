import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
from collections import deque
from plyer import notification

CLOSE_THRESHOLD = 0.20
OPEN_THRESHOLD = 0.24
CLOSED_TIME = 1.5
ALARM_INTERVAL = 2.0
SMOOTH_FRAMES = 5

left_eye = [33, 160, 158, 133, 153, 144]
right_eye = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh
alarm_event = threading.Event()
alarm_enabled = True
alarm_lock = threading.Lock()

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(points, indexes):
    p = [points[i] for i in indexes]
    v1 = distance(p[1], p[5])
    v2 = distance(p[2], p[4])
    h = distance(p[0], p[3])
    return (v1 + v2) / (2 * h) if h else 0


def alarm_worker():
    while True:
        alarm_event.wait()

        with alarm_lock:
            enabled = alarm_enabled

        if alarm_event.is_set() and enabled:
            try:
                winsound.Beep(1200, 350)
            except RuntimeError:
                winsound.MessageBeep()

        time.sleep(ALARM_INTERVAL)


def send_notification():
    try:
        notification.notify(
            title="Warning",
            message="Prolonged eye closure detected. Stop driving safely and rest.",
            app_name="Drowsy-Protector",
            timeout=5
        )
    except Exception as e:
        print("Notification failed:", e)

def reset_monitoring(ear_values):
    ear_values.clear()
    return None, False, False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

ear_values = deque(maxlen=SMOOTH_FRAMES)
closed_start = None
eyes_closed = False
alert = False
last_face_time = time.monotonic()
face_lost = False

threading.Thread(target=alarm_worker, daemon=True).start()

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
        new_alert = False

        if result.multi_face_landmarks:
            last_face_time = now
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
                    new_alert = True
                else:
                    status = "EYES CLOSING"
                    color = (0, 165, 255)

                cv2.putText(
                    frame, f"Closed: {duration:.1f}s",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

            if ear is not None:
                cv2.putText(
                    frame, f"EAR: {ear:.3f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

        else:
            if now - last_face_time > 0.5:
                ear_values.clear()
                eyes_closed = False
                closed_start = None
                status = "FACE NOT DETECTED"
                color = (0, 165, 255)
                face_lost = True

        if new_alert and not alert:
            send_notification()

        alert = new_alert

        if alert and alarm_enabled:
            alarm_event.set()
        else:
            alarm_event.clear()

        cv2.putText(
            frame, status,
            (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )

        alarm_text = "ON" if alarm_enabled else "OFF"
        cv2.putText(
            frame, f"Alarm: {alarm_text} (A to toggle)",
            (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

        if alert:
            cv2.rectangle(frame, (0, 0), (w, 115), (0, 0, 180), 2)
            cv2.putText(
                frame, "WARNING: EYES CLOSED TOO LONG",
                (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        cv2.imshow("DrowsyGuard AI - Phase 3", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("a"):
            with alarm_lock:
                alarm_enabled = not alarm_enabled

        elif key == ord("r"):
            closed_start, eyes_closed, alert = reset_monitoring(ear_values)
            alarm_event.clear()

cap.release()
alarm_event.clear()
cv2.destroyAllWindows()
