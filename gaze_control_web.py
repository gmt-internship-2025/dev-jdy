# gaze_control_web.py 

import cv2
import pickle
import os
import numpy as np
import requests
from gaze_tracking2 import GazeTracking
from my_calibrator import Calibrator

CAM_WIDTH = 960
CAM_HEIGHT = 540
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540

gaze = GazeTracking()
calibrator = Calibrator()

if os.path.exists("calibration_data_amp15.pkl"):
    with open("calibration_data_amp15.pkl", "rb") as f:
        data = pickle.load(f)
        calibrator.X = data["X"]
        calibrator.Y_x = data["Y_x"]
        calibrator.Y_y = data["Y_y"]
        calibrator.reg_x.fit(calibrator.X, calibrator.Y_x)
        calibrator.reg_y.fit(calibrator.X, calibrator.Y_y)
        calibrator.fitted = True
else:
    print("calibration_data_amp15.pkl not found.")
    exit()

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

while True:
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    gaze.refresh(frame)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    if left_pupil and right_pupil:
        pupil_avg = (
            (left_pupil[0] + right_pupil[0]) / 2,
            (left_pupil[1] + right_pupil[1]) / 2
        )
        pupil_norm = (
            pupil_avg[0] / CAM_WIDTH,
            pupil_avg[1] / CAM_HEIGHT
        )
        pupil_amp = (
            (pupil_norm[0] - 0.5) * 15,
            (pupil_norm[1] - 0.5) * 15
        )
        pred_norm = calibrator.predict(pupil_amp)
        pred_norm = np.clip(pred_norm, 0, 1)
        pred_screen = (
            int(pred_norm[0] * SCREEN_WIDTH),
            int(pred_norm[1] * SCREEN_HEIGHT)
        )

        # Flask 서버로 예측값 전송
        try:
            requests.post("http://localhost:5000/api/gaze", json={
                "x": pred_screen[0],
                "y": pred_screen[1]
            })
        except:
            pass

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

