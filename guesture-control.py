import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
import tkinter as tk
from collections import deque

# ---------------- CONFIG ----------------
SCROLL_ANGLE = 15
TURN_ANGLE = 20
COOLDOWN = 1.2
SMOOTHING = 5
# ---------------------------------------

gesture_active = False
app_running = True

last_action_time = 0
pitch_history = deque(maxlen=SMOOTHING)
yaw_history = deque(maxlen=SMOOTHING)

# MediaPipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False)

# 3D face model points
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
])


def gesture_controller():
    global last_action_time, app_running

    cap = cv2.VideoCapture(0)

    while app_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks and gesture_active:
            face = results.multi_face_landmarks[0]

            image_points = np.array([
                (face.landmark[1].x * w, face.landmark[1].y * h),
                (face.landmark[152].x * w, face.landmark[152].y * h),
                (face.landmark[33].x * w, face.landmark[33].y * h),
                (face.landmark[263].x * w, face.landmark[263].y * h),
                (face.landmark[61].x * w, face.landmark[61].y * h),
                (face.landmark[291].x * w, face.landmark[291].y * h)
            ], dtype="double")

            focal_length = w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])

            dist_coeffs = np.zeros((4, 1))

            _, rot_vec, _ = cv2.solvePnP(
                model_points,
                image_points,
                cam_matrix,
                dist_coeffs
            )

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch, yaw, _ = angles

            pitch_history.append(pitch)
            yaw_history.append(yaw)

            avg_pitch = np.mean(pitch_history)
            avg_yaw = np.mean(yaw_history)

            current_time = time.time()

            if current_time - last_action_time > COOLDOWN:
                if avg_pitch > SCROLL_ANGLE:
                    pyautogui.scroll(-300)
                    last_action_time = current_time

                elif avg_pitch < -SCROLL_ANGLE:
                    pyautogui.scroll(300)
                    last_action_time = current_time

                elif avg_yaw > TURN_ANGLE:
                    pyautogui.hotkey('alt', 'right')
                    last_action_time = current_time

                elif avg_yaw < -TURN_ANGLE:
                    pyautogui.hotkey('alt', 'left')
                    last_action_time = current_time

    cap.release()


# ---------------- GUI FUNCTIONS ----------------

def start_gesture():
    global gesture_active
    gesture_active = True
    status_label.config(text="Status: RUNNING", fg="green")


def stop_gesture():
    global gesture_active
    gesture_active = False
    status_label.config(text="Status: STOPPED", fg="red")


def exit_app():
    global app_running
    app_running = False
    root.destroy()


# ---------------- GUI SETUP ----------------

root = tk.Tk()
root.title("Head Gesture Controller")
root.geometry("300x220")
root.resizable(False, False)

title = tk.Label(root, text="Head Gesture Control", font=("Arial", 14, "bold"))
title.pack(pady=10)

status_label = tk.Label(root, text="Status: STOPPED", fg="red", font=("Arial", 11))
status_label.pack(pady=5)

start_btn = tk.Button(root, text="START", width=15, bg="green", fg="white", command=start_gesture)
start_btn.pack(pady=5)

stop_btn = tk.Button(root, text="STOP", width=15, bg="orange", fg="white", command=stop_gesture)
stop_btn.pack(pady=5)

exit_btn = tk.Button(root, text="EXIT", width=15, bg="red", fg="white", command=exit_app)
exit_btn.pack(pady=10)

# Start background thread
threading.Thread(target=gesture_controller, daemon=True).start()

root.mainloop()
#has the start stop and camera_feedback 