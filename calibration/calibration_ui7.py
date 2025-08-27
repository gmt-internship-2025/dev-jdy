# calibration_ui7.py
# 전체화면으로  calibration 하는 화면

import cv2
import time
import numpy as np
import pickle
import tkinter as tk  # 모니터 해상도 가져오기용

from calibration.gaze_tracking2 import GazeTracking
from calibration.my_calibrator import Calibrator

# === 모니터 해상도 자동 가져오기 ===
root = tk.Tk()
root.withdraw()  # Tkinter 창 숨김
MONITOR_WIDTH = root.winfo_screenwidth()
MONITOR_HEIGHT = root.winfo_screenheight()
root.destroy()

# 캘리브레이션 점당 표시 시간 (초)
POINT_DISPLAY_TIME = 3.0

# 웹캠 해상도
CAM_WIDTH, CAM_HEIGHT = 960, 540

# 객체 초기화
gaze = GazeTracking()
calibrator = Calibrator()

# 웹캠 열기
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# === 전체화면 모드로 창 표시 ===
cv2.namedWindow("Calibration UI", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration UI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # 현재 캘리브레이션 타겟 위치 (픽셀 좌표)
    target = calibrator.getCurrentPoint(MONITOR_WIDTH, MONITOR_HEIGHT)  # 모니터 해상도 기준
    target = tuple(map(int, target))

    # pupil 좌표 수집 리스트
    collected_pupil = []

    start_time = time.time()
    while time.time() - start_time < POINT_DISPLAY_TIME:
        _, frame = webcam.read()

        # 좌우 반전
        frame = cv2.flip(frame, 1)

        gaze.refresh(frame)

        if gaze.pupils_located:
            pupil = gaze.pupil_left_coords()
            if pupil:
                collected_pupil.append(pupil)  # 정규화 없이 그대로 저장

        # 타겟 점 표시 (전체화면 크기 기준)
        dot_frame = cv2.resize(frame, (MONITOR_WIDTH, MONITOR_HEIGHT))
        cv2.circle(dot_frame, target, 20, (0, 255, 0), -1)
        cv2.imshow("Calibration UI", dot_frame)

        if cv2.waitKey(1) == 27:
            webcam.release()
            cv2.destroyAllWindows()
            exit()

    if collected_pupil:
        pupil_array = np.array(collected_pupil)
        pupil_mean = np.mean(pupil_array, axis=0)

        # pupil 좌표 정규화 (0~1 스케일, 카메라 해상도 기준)
        pupil_norm = (
            pupil_mean[0] / CAM_WIDTH,
            pupil_mean[1] / CAM_HEIGHT
        )

        # 민감도 증가
        pupil_amp = (
            (pupil_norm[0] - 0.5) * 15,
            (pupil_norm[1] - 0.5) * 15
        )

        # target 좌표 정규화 (모니터 해상도 기준)
        target_norm = (
            target[0] / MONITOR_WIDTH,
            target[1] / MONITOR_HEIGHT
        )

        print(f"→ pupil norm: {pupil_norm}")
        print(f"→ pupil amp: {pupil_amp}")
        print(f"→ target norm: {target_norm}")

        calibrator.add(pupil_amp, target_norm)
        print(f"[+] Collected {len(collected_pupil)} samples at {target}")

    calibrator.movePoint()

    if calibrator.matrix.iterator == 0:
        print("Calibration Complete!")
        break

webcam.release()
cv2.destroyAllWindows()

# 학습 데이터 저장
with open("calibration_data1.pkl", "wb") as f:
    pickle.dump({
        "X": calibrator.X,
        "Y_x": calibrator.Y_x,
        "Y_y": calibrator.Y_y
    }, f)

print("학습 데이터 calibration_data1.pkl 저장 완료")

