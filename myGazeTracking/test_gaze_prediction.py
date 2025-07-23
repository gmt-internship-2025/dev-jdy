import cv2
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "GazeTracking")))

from gaze_tracking import GazeTracking
from calibration_mapping import GazeMapper

WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720

# 정규화 함수
def normalize_coords(coords, face_box):
    (fx, fy, fw, fh) = face_box
    if fw == 0 or fh == 0:  # 안전 처리
        return (0.5, 0.5)
    nx = (coords[0] - fx) / fw
    ny = (coords[1] - fy) / fh
    return (nx, ny)

def main():
    gaze = GazeTracking()
    mapper = GazeMapper()
    mapper.load("gaze_model.pkl")
    if not mapper.is_trained:
        print("[ERROR] 모델 없음. 먼저 calibration 실행 필요.")
        return

    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    if not webcam.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    print("[INFO] 실시간 시선 예측 시작...")

    while True:
        ret, frame = webcam.read()
        if not ret or frame is None:
            print("[WARN] 프레임 읽기 실패")
            # 화면은 그래도 띄우기
            cv2.imshow("Gaze Prediction", np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8))
            if cv2.waitKey(1) == 27:
                break
            continue

        gaze.refresh(frame)

        face_box = None
        if gaze.eye_left is not None and gaze.eye_right is not None:
            xs = gaze.eye_left.landmark_points[:,0].tolist() + gaze.eye_right.landmark_points[:,0].tolist()
            ys = gaze.eye_left.landmark_points[:,1].tolist() + gaze.eye_right.landmark_points[:,1].tolist()
            fx, fy, fw, fh = min(xs), min(ys), (max(xs)-min(xs)), (max(ys)-min(ys))
            if fw > 0 and fh > 0:  # 안전 처리
                face_box = (fx, fy, fw, fh)

        # pupil 좌표
        coords = None
        left = gaze.pupil_left_coords()
        right = gaze.pupil_right_coords()
        if left is not None and right is not None:
            coords = np.mean([left, right], axis=0)
        elif left is not None:
            coords = left
        elif right is not None:
            coords = right

        if coords is not None and face_box is not None:
            norm_coords = normalize_coords(coords, face_box)
            pred_x, pred_y = mapper.predict(norm_coords)
            # 범위 체크
            px = max(0, min(WINDOW_WIDTH-1, int(pred_x)))
            py = max(0, min(WINDOW_HEIGHT-1, int(pred_y)))
            cv2.circle(frame, (px, py), 10, (0,0,255), -1)
            cv2.putText(frame, f"Gaze: ({px},{py})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # face_box가 없어도 항상 프레임 출력
        cv2.imshow("Gaze Prediction", frame)

        # waitKey는 항상 호출
        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

