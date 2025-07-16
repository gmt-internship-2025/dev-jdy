# test_gaze_prediction.py
import cv2
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "GazeTracking")))

from gaze_tracking import GazeTracking
from calibration_mapping import GazeMapper

WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

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

    print("[INFO] 실시간 시선 예측 시작...")

    while True:
        _, frame = webcam.read()
        gaze.refresh(frame)

        coords = gaze.pupil_left_coords()
        if coords and mapper.is_trained:
            pred_x, pred_y = mapper.predict(coords)
            cv2.circle(frame, (pred_x, pred_y), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Gaze: ({pred_x}, {pred_y})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gaze Prediction", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

