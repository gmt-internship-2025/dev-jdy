# gaze_control7.py
# 예측 시 pupil 증폭 적용

import cv2
import pyautogui
import pickle
import os
import numpy as np
from gaze_tracking2 import GazeTracking
from my_calibrator import Calibrator

# 화면 해상도
SCREEN_WIDTH, SCREEN_HEIGHT = 960, 540

# 웹캠 해상도
CAM_WIDTH = 960
CAM_HEIGHT = 540

# 객체 초기화
gaze = GazeTracking()
calibrator = Calibrator()

# [수정] 학습 데이터 불러오기 + 학습 확인 출력
if os.path.exists("calibration_data.pkl"):
    with open("calibration_data.pkl", "rb") as f:
        data = pickle.load(f)
        calibrator.X = data["X"]
        calibrator.Y_x = data["Y_x"]
        calibrator.Y_y = data["Y_y"]
        
        print(f"[+] Loaded {len(calibrator.X)} samples")

        calibrator.reg_x.fit(calibrator.X, calibrator.Y_x)
        calibrator.reg_y.fit(calibrator.X, calibrator.Y_y)
        calibrator.fitted = True

        print("학습 데이터 불러오기 및 모델 학습 완료")
else:
    print("[!] calibration_data.pkl 파일이 없습니다.")
    exit()

# 웹캠 시작
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

while True:
    _, frame = webcam.read()
    
    # [수정6] 좌우 반전 추가
    frame = cv2.flip(frame, 1)
    
    gaze.refresh(frame)

    # 동공 좌표
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    if left_pupil and right_pupil:
        # 평균 좌표
        pupil_avg = (
            (left_pupil[0] + right_pupil[0]) / 2,
            (left_pupil[1] + right_pupil[1]) / 2
        )

        ## [수정] 정규화
        # pupil_norm = (
        #    pupil_avg[0] / CAM_WIDTH,
        #    pupil_avg[1] / CAM_HEIGHT
        # )
        
        # [수정] 예측 수행 (정규화 입력)
        # pred_norm = calibrator.predict(pupil_norm)

        # [수정] 학습 때 raw 좌표 (600~630)이었으니, 그대로 예측에도 넣기
        # pupil_input = pupil_avg
        # pred_norm = calibrator.predict(pupil_input)
        
        # [수정] 예측값 보정
        # pred_norm = np.clip(pred_norm, 0, 1)
        
        # 입력도 0~1 스케일로 정규화해서 예측
        pupil_norm = (
            pupil_avg[0] / CAM_WIDTH,
            pupil_avg[1] / CAM_HEIGHT
        )
        
        # 정규화된 pupil 좌표 확대 (중앙 기준 ±2.0 스케일)
        pupil_amp = (
            (pupil_norm[0] - 0.5) * 100,
            (pupil_norm[1] - 0.5) * 100
        )
        
        # 예측 수행
        pred_norm = calibrator.predict(pupil_amp)
        
        # 예측값 튀는 것 방지
        pred_norm = np.clip(pred_norm, 0, 1)
        
        if not (0 <= pred_norm[0] <= 1 and 0 <= pred_norm[1] <= 1):
            print("예측값이 정규화 범위를 벗어났습니다:", pred_norm)
        
        # [수정] 화면 해상도로 복원
        pred_screen = (
            int(pred_norm[0] * SCREEN_WIDTH),
            int(pred_norm[1] * SCREEN_HEIGHT)
        )

        # [수정] 디버깅 출력
        print(f"← raw left: {left_pupil}, raw right: {right_pupil}")
        print(f"→ pupil norm input: {pupil_norm}")
        print(f"→ pupil amp input: {pupil_amp}")
        print(f"→ pred norm: {pred_norm}")
        print(f"→ pred screen: {pred_screen}\n")

        # 마우스 이동
        # if 0 <= pred_screen[0] <= SCREEN_WIDTH and 0 <= pred_screen[1] <= SCREEN_HEIGHT:
        #    pyautogui.moveTo(pred_screen[0], pred_screen[1], duration=0.1)

    # 디버그 화면
    annotated = gaze.annotated_frame()
    if left_pupil and right_pupil:
        cv2.putText(annotated, f"Pupil: {pupil_avg}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Pred: {pred_screen}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        # 예측 위치 시각화 (파란 원)
        cv2.circle(annotated, pred_screen, 15, (255, 0, 0), 2)

    cv2.imshow("Gaze Mouse Control", annotated)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

