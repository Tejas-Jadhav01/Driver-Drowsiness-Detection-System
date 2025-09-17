import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from detection_utils import calculate_ear, calculate_mar, estimate_head_pose

# --- Constants ---
EAR_THRESHOLD = 0.25
EYE_CLOSURE_DURATION_THRESHOLD = 1.5  # Reduced for better demo purposes
MAR_THRESHOLD = 0.5
HEAD_PITCH_THRESHOLD = 20.0 # Adjusted for more noticeable nod detection

class DrowsinessApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.drowsy_alert_on = False
        self.eye_closure_start_time = None

        # --- Initialize Detection Models ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Initialize Pygame for Audio ---
        pygame.mixer.init()
        pygame.mixer.music.load("assets/alert.wav")

        # --- Video Capture ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera.")
            return

        # --- GUI Setup ---
        self.create_widgets()

        # --- Start the update loop ---
        self.update()

    def create_widgets(self):
        # --- Main Frame ---
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Video Canvas ---
        self.canvas = tk.Canvas(main_frame, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=0, column=0, columnspan=2)

        # --- Status Labels ---
        status_frame = ttk.LabelFrame(main_frame, text="Detection Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.ear_label = ttk.Label(status_frame, text="EAR: N/A", font=("Helvetica", 12))
        self.ear_label.grid(row=0, column=0, padx=10, sticky=tk.W)

        self.mar_label = ttk.Label(status_frame, text="MAR: N/A", font=("Helvetica", 12))
        self.mar_label.grid(row=0, column=1, padx=10, sticky=tk.W)
        
        self.pitch_label = ttk.Label(status_frame, text="Pitch: N/A", font=("Helvetica", 12))
        self.pitch_label.grid(row=0, column=2, padx=10, sticky=tk.W)

        self.alert_label = ttk.Label(status_frame, text="Status: Awake", font=("Helvetica", 14, "bold"), foreground="green")
        self.alert_label.grid(row=1, column=0, columnspan=3, pady=10)

    def play_alert_sound(self):
        while self.drowsy_alert_on:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

    def update(self):
        success, image = self.cap.read()
        if not success:
            self.window.after(10, self.update)
            return

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        image.flags.writeable = True

        alert_text = "Status: Awake"
        alert_color = "green"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                
                # --- EAR Calculation ---
                left_eye_points = np.array([[landmarks[i][0], landmarks[i][1]] for i in LEFT_EYE_EAR_INDICES])
                right_eye_points = np.array([[landmarks[i][0], landmarks[i][1]] for i in RIGHT_EYE_EAR_INDICES])
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0
                self.ear_label.config(text=f"EAR: {avg_ear:.2f}")

                # --- Drowsiness Detection ---
                if avg_ear < EAR_THRESHOLD:
                    if self.eye_closure_start_time is None:
                        self.eye_closure_start_time = time.time()
                    else:
                        closure_duration = time.time() - self.eye_closure_start_time
                        if closure_duration >= EYE_CLOSURE_DURATION_THRESHOLD:
                            if not self.drowsy_alert_on:
                                self.drowsy_alert_on = True
                                with open("logs/drowsiness_log.txt", "a") as log:
                                    log.write(f"Drowsiness detected at {time.ctime()}\n")
                                alert_thread = threading.Thread(target=self.play_alert_sound)
                                alert_thread.daemon = True
                                alert_thread.start()
                            alert_text = "DROWSINESS ALERT!"
                            alert_color = "red"
                else:
                    self.eye_closure_start_time = None
                    if self.drowsy_alert_on:
                        self.drowsy_alert_on = False
                        pygame.mixer.music.stop()

                # --- Yawn Detection ---
                mouth_points = np.array([[landmarks[i][0], landmarks[i][1]] for i in MOUTH_MAR_INDICES])
                mar = calculate_mar(mouth_points)
                self.mar_label.config(text=f"MAR: {mar:.2f}")
                if mar > MAR_THRESHOLD:
                    alert_text = "YAWN DETECTED"
                    alert_color = "orange"

                # --- Head Pose Estimation ---
                image_shape = image.shape
                landmarks_2d = np.array([(landmarks[i][0] * image_shape[1], landmarks[i][1] * image_shape[0]) for i in HEAD_POSE_LANDMARKS], dtype="double")
                pitch, _, _, nose_end_point2D = estimate_head_pose(landmarks_2d, image_shape)
                self.pitch_label.config(text=f"Pitch: {pitch:.2f}")
                if pitch > HEAD_PITCH_THRESHOLD:
                    alert_text = "HEAD NOD"
                    alert_color = "orange"
                
                # Draw head pose line
                p1 = (int(landmarks_2d[0][0]), int(landmarks_2d[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.line(image, p1, p2, (255, 0, 0), 2)

        self.alert_label.config(text=alert_text, foreground=alert_color)

        # --- Display Video ---
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

    def on_closing(self):
        self.face_mesh.close()
        self.cap.release()
        self.window.destroy()

# --- Landmark Indices ---
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_MAR_INDICES = [13, 14, 78, 308]
HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]

# --- Main Execution ---
if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    import os
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    root = tk.Tk()
    app = DrowsinessApp(root, "Driver Drowsiness Detection System")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
