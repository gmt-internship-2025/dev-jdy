## calibration_ui4.py

import cv2
import time
import numpy as np
from gaze_tracking2 import GazeTracking
from my_calibrator import Calibrator
import pickle

# 캘리브레이션 점당 표시 시간 (초)
POINT_DISPLAY_TIME = 2.0

# 웹캠 해상도
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# 객체 초기화
gaze = GazeTracking()
calibrator = Calibrator()

# 웹캠 열기
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

while True:
    # 현재 캘리브레이션 타겟 위치 (픽셀 좌표)
    target = calibrator.getCurrentPoint(CAM_WIDTH, CAM_HEIGHT)
    target = tuple(map(int, target))

    # pupil 좌표 수집 리스트
    collected_pupil = []

    start_time = time.time()
    while time.time() - start_time < POINT_DISPLAY_TIME:
        _, frame = webcam.read()
        
        # [수정4] 좌우 반전 추가
        frame = cv2.flip(frame, 1)
        
        gaze.refresh(frame)

        if gaze.pupils_located:
            pupil = gaze.pupil_left_coords()
            if pupil:
                collected_pupil.append(pupil)  # 정규화 없이 그대로 저장

        # 타겟 점 표시
        dot_frame = frame.copy()
        cv2.circle(dot_frame, target, 20, (0, 255, 0), -1)
        cv2.imshow("Calibration UI", dot_frame)

        if cv2.waitKey(1) == 27:
            webcam.release()
            cv2.destroyAllWindows()
            exit()

    if collected_pupil:
        pupil_array = np.array(collected_pupil)
        pupil_mean = np.mean(pupil_array, axis=0)

        # target 좌표 정규화
        # target_norm = (
        #     target[0] / CAM_WIDTH,
        #     target[1] / CAM_HEIGHT
        # )

        # pupil 좌표 정규화 (입력도 0~1 스케일로 변경)
        pupil_norm = (
            pupil_mean[0] / CAM_WIDTH,
            pupil_mean[1] / CAM_HEIGHT
        )
        
        # target 좌표 정규화
        target_norm = (
            target[0] / CAM_WIDTH,
            target[1] / CAM_HEIGHT
        )

        print(f"→ pupil norm: {pupil_norm}")
        print(f"→ target norm: {target_norm}")

        # calibrator.add(pupil_mean, target_norm)
        calibrator.add(pupil_norm, target_norm)
        print(f"[+] Collected {len(collected_pupil)} samples at {target}")

    calibrator.movePoint()

    if calibrator.matrix.iterator == 0:
        print("Calibration Complete!")
        break

webcam.release()
cv2.destroyAllWindows()

# 학습 데이터 저장
with open("calibration_data.pkl", "wb") as f:
    pickle.dump({
        "X": calibrator.X,
        "Y_x": calibrator.Y_x,
        "Y_y": calibrator.Y_y
    }, f)

print("학습 데이터 calibration_data.pkl 저장 완료")

