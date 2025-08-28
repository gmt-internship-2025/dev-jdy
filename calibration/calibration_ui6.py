# calibration_ui6.py (Jetson Orin)

import cv2
import time
import numpy as np
from calibration.gaze_tracking2 import GazeTracking
from calibration.my_calibrator import Calibrator
import pickle
from pathlib import Path
import conf # 설정 파일 import

# 객체 초기화
gaze = GazeTracking()
calibrator = Calibrator()

# 웹캠 열기 (conf.py 설정값 사용)
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, conf.CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, conf.CAM_HEIGHT)
webcam.set(cv2.CAP_PROP_FPS, conf.CAM_FPS)

# ==== 저장 경로를 프로젝트 루트 기준으로 설정 (conf.py 설정값 사용) ====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = PROJECT_ROOT / conf.CALIBRATION_FILE

while True:
    # conf.py의 카메라 해상도 값을 사용
    target = calibrator.getCurrentPoint(conf.CAM_WIDTH, conf.CAM_HEIGHT)
    target = tuple(map(int, target))

    collected_pupil = []
    start_time = time.time()
    # conf.py의 점 응시 시간 값을 사용
    while time.time() - start_time < conf.POINT_DISPLAY_TIME:
        ok, frame = webcam.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        gaze.refresh(frame)

        if gaze.pupils_located:
            pupil = gaze.pupil_left_coords()
            if pupil:
                collected_pupil.append(pupil)

        dot_frame = frame.copy()
        cv2.circle(dot_frame, target, 20, (0, 255, 0), -1)
        cv2.imshow("Calibration UI", dot_frame)

        if cv2.waitKey(1) == 27:
            webcam.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)

    if collected_pupil:
        pupil_array = np.array(collected_pupil)
        pupil_mean = np.mean(pupil_array, axis=0)

        # conf.py의 카메라 해상도 값을 사용
        pupil_norm = (pupil_mean[0] / conf.CAM_WIDTH, pupil_mean[1] / conf.CAM_HEIGHT)
        pupil_amp = ((pupil_norm[0] - 0.5) * 15, (pupil_norm[1] - 0.5) * 15)
        target_norm = (target[0] / conf.CAM_WIDTH, target[1] / conf.CAM_HEIGHT)

        print(f"→ pupil norm: {pupil_norm}")
        print(f"→ pupil amp: {pupil_amp}")
        print(f"→ target norm: {target_norm}")

        calibrator.add(pupil_amp, target_norm)
        print(f"[+] Collected {len(collected_pupil)} samples at {target}\n")

    calibrator.movePoint()

    if calibrator.matrix.iterator == 0:
        print("Calibration Complete!")
        break

webcam.release()
cv2.destroyAllWindows()

# ==== 절대 경로로 저장 ====
with open(CALIB_PATH, "wb") as f:
    pickle.dump({"X": calibrator.X, "Y_x": calibrator.Y_x, "Y_y": calibrator.Y_y}, f)
print(f"학습 데이터 저장 완료: {CALIB_PATH}")

