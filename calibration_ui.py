# calibration_ui.py

import cv2
import time
import numpy as np
from gaze_tracking2 import GazeTracking
from my_calibrator import Calibrator

# 캘리브레이션 점 정보
POINT_DISPLAY_TIME = 2.0  # 초
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720

# GazeTracking & Calibrator 초기화
gaze = GazeTracking()
calibrator = Calibrator()

# 웹캠 시작
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

# 메인 루프
while True:
    # 현재 캘리브레이션할 점 좌표
    target = calibrator.getCurrentPoint(WINDOW_WIDTH, WINDOW_HEIGHT)
    target = tuple(map(int, target))
    
    # 좌표 수집을 위한 임시 저장소
    collected_pupil = []

    start_time = time.time()
    while time.time() - start_time < POINT_DISPLAY_TIME:
        _, frame = webcam.read()
        gaze.refresh(frame)

        # pupil 좌표 수집
        if gaze.pupils_located:
            pupil = gaze.pupil_left_coords()
            if pupil:
                collected_pupil.append(pupil)

        # 점 그리기
        dot_frame = frame.copy()
        cv2.circle(dot_frame, target, 20, (0, 255, 0), -1)
        cv2.imshow("Calibration UI", dot_frame)

        if cv2.waitKey(1) == 27:
            webcam.release()
            cv2.destroyAllWindows()
            exit()

    # pupil 좌표 평균 계산
    if collected_pupil:
        pupil_array = np.array(collected_pupil)
        pupil_mean = np.mean(pupil_array, axis=0)
        calibrator.add(pupil_mean, target)
        print(f"[+] Collected {len(collected_pupil)} samples at {target}")

    # 다음 점으로 이동
    calibrator.movePoint()

    # 모든 점 끝났는지 확인
    if calibrator.matrix.iterator == 0:
        print(" Calibration Complete!")
        break

webcam.release()
cv2.destroyAllWindows()

