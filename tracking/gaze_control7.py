# gaze_control7.py
# 예측 시 pupil 증폭 적용 + CSV 로깅

import cv2
import pyautogui
import pickle
import os
import numpy as np
import csv
from calibration.gaze_tracking2 import GazeTracking
from calibration.my_calibrator import Calibrator

# 화면 해상도
SCREEN_WIDTH, SCREEN_HEIGHT = 1840, 960

# 웹캠 해상도
CAM_WIDTH = 1840
CAM_HEIGHT = 960   # ← 오타(CAM_EIGHT) 수정

# 객체 초기화
gaze = GazeTracking()
calibrator = Calibrator()

# [수정] 학습 데이터 불러오기 + 학습 확인 출력
if os.path.exists("calibration_data_1.pkl"):
    with open("calibration_data_1.pkl", "rb") as f:
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
    print("[!] calibration_data_1.pkl 파일이 없습니다.")
    exit()

# === CSV 로그 초기화 ===
log_file = "accuracy_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pupil_norm_x", "pupil_norm_y",
            "pupil_amp_x", "pupil_amp_y",
            "pred_norm_x", "pred_norm_y",
            "pred_screen_x", "pred_screen_y"
        ])

# 웹캠 시작
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

while True:
    ret, frame = webcam.read()
    if not ret:
        print("[!] 카메라 프레임을 읽을 수 없습니다.")
        break
    
    # 좌우 반전
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
        
        # 정규화
        pupil_norm = (
            pupil_avg[0] / CAM_WIDTH,
            pupil_avg[1] / CAM_HEIGHT
        )
        
        # 증폭
        pupil_amp = (
            (pupil_norm[0] - 0.5) * 15,
            (pupil_norm[1] - 0.5) * 15
        )
        
        # 예측
        pred_norm = calibrator.predict(pupil_amp)
        pred_norm = np.clip(pred_norm, 0, 1)
        
        # 화면 좌표로 변환
        pred_screen = (
            int(pred_norm[0] * SCREEN_WIDTH),
            int(pred_norm[1] * SCREEN_HEIGHT)
        )
        
        # === CSV 저장 ===
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                pupil_norm[0], pupil_norm[1],
                pupil_amp[0], pupil_amp[1],
                pred_norm[0], pred_norm[1],
                pred_screen[0], pred_screen[1]
            ])

        # === 디버깅 출력 ===
        print(f"← raw left: {left_pupil}, raw right: {right_pupil}")
        print(f"→ pupil norm input: {pupil_norm}")
        print(f"→ pupil amp input: {pupil_amp}")
        print(f"→ pred norm: {pred_norm}")
        print(f"→ pred screen: {pred_screen}\n")

        # 마우스 이동 (옵션)
        # if 0 <= pred_screen[0] <= SCREEN_WIDTH and 0 <= pred_screen[1] <= SCREEN_HEIGHT:
        #     pyautogui.moveTo(pred_screen[0], pred_screen[1], duration=0.1)

    # 디버그 화면
    annotated = gaze.annotated_frame()
    if left_pupil and right_pupil:
        cv2.putText(annotated, f"Pupil: {pupil_avg}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Pred: {pred_screen}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        cv2.circle(annotated, pred_screen, 15, (255, 0, 0), 2)

    cv2.imshow("Gaze Mouse Control", annotated)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

