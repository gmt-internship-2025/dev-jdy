import cv2
import time
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "GazeTracking")))

from gaze_tracking import GazeTracking
from calibration_mapping import GazeMapper

WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720

def main():
    gaze = GazeTracking()
    mapper = GazeMapper()

    # 모델 불러오기
    mapper.load("gaze_model.pkl")
    if not mapper.is_trained:
        print("[ERROR] GazeMapper가 아직 학습되지 않았습니다. 먼저 test_gaze_calibration.py를 실행하세요.")
        return

    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    if not webcam.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다. 장치 번호를 확인하세요.")
        return

    print("[INFO] 실시간 시선 예측 시작...")

    while True:
        ret, frame = webcam.read()
        if not ret or frame is None:
            print("[WARN] 프레임을 읽지 못했습니다.")
            continue

        gaze.refresh(frame)

        # [수정] 양안 평균 계산
        left = gaze.pupil_left_coords()
        right = gaze.pupil_right_coords()
        if left is not None and right is not None:
            coords = np.mean([left, right], axis=0)
        elif left is not None:
            coords = left
        elif right is not None:
            coords = right
        else:
            coords = None

        if coords is not None and mapper.is_trained:
            pred_x, pred_y = mapper.predict(coords)

            print(f"[DEBUG] pred_x={pred_x}, pred_y={pred_y}")

            px = int(pred_x)
            py = int(pred_y)

            px = max(0, min(WINDOW_WIDTH - 1, px))
            py = max(0, min(WINDOW_HEIGHT - 1, py))

            cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Gaze: ({px}, {py})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gaze Prediction", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

