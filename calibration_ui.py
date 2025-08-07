# calibration_ui2.py

import cv2
import time
import numpy as np
from gaze_tracking2 import GazeTracking
from my_calibrator import Calibrator

# [수정2] 캘리브레이션 점 정보
POINT_DISPLAY_TIME = 2.0  # 초 동안 pupil 좌표 수집
CAM_WIDTH, CAM_HEIGHT = 1280, 720  # 웹캠 해상도

# GazeTracking & Calibrator 초기화
gaze = GazeTracking()
calibrator = Calibrator()

# [수정2] 웹캠 시작: CAM_WIDTH
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# 메인 루프
while True:
    # 현재 캘리브레이션할 점 좌표
    target = calibrator.getCurrentPoint(CAM_WIDTH, CAM_HEIGHT)
    target = tuple(map(int, target))
    
    # 좌표 수집을 위한 임시 저장소
    collected_pupil = []

    start_time = time.time()
    while time.time() - start_time < POINT_DISPLAY_TIME:
        _, frame = webcam.read()
        gaze.refresh(frame)

        # [수정2] pupil 좌표 수집 - 정규화 -> 불필요한 정규화 제거
        if gaze.pupils_located:
            pupil = gaze.pupil_left_coords()
            if pupil:
            #    pupil_norm = (
            #        pupil[0] / CAM_WIDTH,
            #        pupil[1] / CAM_HEIGHT
            #    ) 
                collected_pupil.append(pupil)

        # 점 그리기
        dot_frame = frame.copy()
        cv2.circle(dot_frame, target, 20, (0, 255, 0), -1)
        cv2.imshow("Calibration UI", dot_frame)

        if cv2.waitKey(1) == 27:
            webcam.release()
            cv2.destroyAllWindows()
            exit()

    # [수정2] pupil 좌표 평균 계산 - 정규화
    if collected_pupil:
        pupil_array = np.array(collected_pupil)
        pupil_mean = np.mean(pupil_array, axis=0)
        
        # pupil_norm = (
        #    pupil_mean[0] / CAM_WIDTH,
        #    pupil_mean[1] / CAM_HEIGHT
        # )
        # print(f"→ pupil raw: {pupil_mean}, norm: {pupil_norm}")
        print(f"→ pupil raw: {pupil_mean}")
        
        # calibrator.add(pupil_mean, target)
        # 수정된 코드 (target을 0~1로 정규화)
        target_norm = (
            target[0] / CAM_WIDTH,
            target[1] / CAM_HEIGHT
        )
        calibrator.add(pupil_mean, target_norm)
        print(f"[+] Collected {len(collected_pupil)} samples at {target}")

    # 다음 점으로 이동
    calibrator.movePoint()

    # 모든 점 끝났는지 확인
    if calibrator.matrix.iterator == 0:
        print(" Calibration Complete!")
        break

webcam.release()
cv2.destroyAllWindows()

# [수정1] 저장 추가
import pickle
with open("calibration_data.pkl", "wb") as f:
    pickle.dump({
        "X": calibrator.X,
        "Y_x": calibrator.Y_x,
        "Y_y": calibrator.Y_y
    }, f)
print("학습 데이터 calibration_data.pkl 저장 완료")
