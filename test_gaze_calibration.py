# test_gaze_calibration.py
import cv2
import time
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "GazeTracking")))

from gaze_tracking import GazeTracking
from calibration_mapping import GazeMapper

# 화면 사이즈 및 점 위치 정의
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
GRID_ROWS, GRID_COLS = 3, 3
DOT_RADIUS = 10
DISPLAY_TIME = 2  # 각 점 응시 시간 (초)

# 3x3 점 생성
def generate_grid_points():
    points = []
    step_x = WINDOW_WIDTH // (GRID_COLS + 1)
    step_y = WINDOW_HEIGHT // (GRID_ROWS + 1)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = step_x * (col + 1)
            y = step_y * (row + 1)
            points.append((x, y))
    return points

def main():
    gaze = GazeTracking()
    mapper = GazeMapper()
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    screen_points = generate_grid_points()  # 9개
    pupil_points = []

    print("[INFO] 시선 보정 시작: 각 점을 응시해 주세요")

    for idx, screen_pt in enumerate(screen_points):
        current_pupil_coords = []
        start_time = time.time()

        while time.time() - start_time < DISPLAY_TIME:
            _, frame = webcam.read()
            gaze.refresh(frame)

            display_frame = frame.copy()
            cv2.circle(display_frame, screen_pt, DOT_RADIUS, (0, 0, 255), -1)
            cv2.putText(display_frame, f"Point {idx+1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration", display_frame)

            coords = gaze.pupil_left_coords()
            if coords:
                current_pupil_coords.append(coords)  # 점별로 모으기

            if cv2.waitKey(1) == 27:  # ESC
                break

        # 한 점에 대해 평균 pupil 좌표만 저장
        if current_pupil_coords:
            avg_coords = np.mean(current_pupil_coords, axis=0)
            pupil_points.append(tuple(avg_coords))
            print(f"[INFO] Point {idx+1} 수집 완료: 평균 좌표 {avg_coords}")
        else:
            print(f"[WARN] Point {idx+1} 에 대해 pupil 좌표 없음")

    webcam.release()
    cv2.destroyAllWindows()

    # 총 9개 수집 확인 후 학습
    if len(pupil_points) == len(screen_points):
        print("[INFO] 시선 매핑 학습 시작...")
        mapper.train(pupil_points, screen_points)
    else:
        print(f"[ERROR] 매핑 학습 실패: {len(pupil_points)}/9개만 수집됨")

if __name__ == "__main__":
    main()

