#python sleep_detector.py - -shape - predictor shape_predictor_68_face_landmarks.dat

# ================================
# IMPORTS
# ================================
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from pygame import mixer

# ================================
# CONFIG
# ================================
dirname = os.path.dirname(__file__)

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 25

COUNTER = 0
TOTAL = 0
SLEEPING = False

# ================================
# FUNCTIONS
# ================================
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ================================
# ARGUMENTS
# ================================
ap = argparse.ArgumentParser()
ap.add_argument(
    "-p",
    "--shape-predictor",
    default="shape_predictor_68_face_landmarks.dat",
    help="path to facial landmark predictor"
)
args = vars(ap.parse_args())

# ================================
# LOAD MODELS
# ================================
print("[INFO] loading facial landmark predictor...")

if not os.path.exists(args["shape_predictor"]):
    raise FileNotFoundError("Shape predictor file not found!")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ================================
# AUDIO SETUP
# ================================
try:
    mixer.init()
    alarm_path = os.path.join(dirname, "alarm.mp3")
    mixer.music.load(alarm_path)
except Exception as e:
    print("[WARNING] Audio error:", e)

# ================================
# CAMERA SETUP
# ================================
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

time.sleep(1.0)

# ================================
# MAIN LOOP
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # draw eyes
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            COUNTER = 0
            SLEEPING = False

        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            SLEEPING = True
            TOTAL += 1
            COUNTER = 0

        # alarm
        if SLEEPING:
            if not mixer.music.get_busy():
                mixer.music.play()
        else:
            mixer.music.pause()

        # UI
        cv2.putText(frame, f"Sleeping: {SLEEPING}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Total Sleeps: {TOTAL}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================================
# CLEANUP
# ================================
cap.release()
cv2.destroyAllWindows()