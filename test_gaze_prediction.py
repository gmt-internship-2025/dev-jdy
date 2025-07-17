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

        coords = gaze.pupil_left_coords()
        if coords and mapper.is_trained:
            pred_x, pred_y = mapper.predict(coords)

            # 디버깅 출력
            print(f"[DEBUG] pred_x={pred_x}, pred_y={pred_y}")

            # 정수형 변환
            px = int(pred_x)
            py = int(pred_y)

            # 화면 범위를 벗어나지 않도록 클램핑
            px = max(0, min(WINDOW_WIDTH - 1, px))
            py = max(0, min(WINDOW_HEIGHT - 1, py))

            # 예측 위치에 빨간 점 표시
            cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)

            # 예측 좌표 텍스트 표시
            cv2.putText(frame, f"Gaze: ({px}, {py})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gaze Prediction", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

