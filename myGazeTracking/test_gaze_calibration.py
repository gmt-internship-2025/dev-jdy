import cv2
import time
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "GazeTracking")))

from gaze_tracking import GazeTracking
from calibration_mapping import GazeMapper

WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
GRID_ROWS, GRID_COLS = 5, 5
DOT_RADIUS = 10
DISPLAY_TIME = 3

# 정규화 함수
def normalize_coords(coords, face_box):
    (fx, fy, fw, fh) = face_box
    if fw == 0 or fh == 0:
        return (0.5, 0.5)  # 기본값을 리턴해서 ZeroDivision 방지
    nx = (coords[0] - fx) / fw
    ny = (coords[1] - fy) / fh
    return (nx, ny)

def main():
    gaze = GazeTracking()
    mapper = GazeMapper()
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    if not webcam.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    # 5초 카운트다운 표시 
    for countdown in range(5, 0, -1):  # 5,4,3,2,1
        ret, frame = webcam.read()
        if not ret or frame is None:
            continue
        # 중앙에 카운트 숫자 표시
        text = str(countdown)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
        x = (WINDOW_WIDTH - text_width) // 2
        y = (WINDOW_HEIGHT + text_height) // 2
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1000) == 27:  # ESC로 중단 가능
            webcam.release()
            cv2.destroyAllWindows()
            return
    # 카운트다운 끝, 보정 시작

    # 그리드 생성
    step_x = WINDOW_WIDTH // (GRID_COLS + 1)
    step_y = WINDOW_HEIGHT // (GRID_ROWS + 1)
    screen_points = [(step_x*(c+1), step_y*(r+1)) for r in range(GRID_ROWS) for c in range(GRID_COLS)]

    pupil_points = []

    print("[INFO] 시선 보정 시작: 각 점을 응시해 주세요")

    for idx, screen_pt in enumerate(screen_points):
        current_pupil_coords = []
        start_time = time.time()

        while time.time() - start_time < DISPLAY_TIME:
            ret, frame = webcam.read()
            if not ret or frame is None:
                print("[WARN] 프레임 읽기 실패")
                continue

            gaze.refresh(frame)

            # 얼굴 박스 계산
            face_box = None
            if gaze.eye_left is not None and gaze.eye_right is not None:
                xs = gaze.eye_left.landmark_points[:,0].tolist() + gaze.eye_right.landmark_points[:,0].tolist()
                ys = gaze.eye_left.landmark_points[:,1].tolist() + gaze.eye_right.landmark_points[:,1].tolist()
                fx, fy, fw, fh = min(xs), min(ys), (max(xs)-min(xs)), (max(ys)-min(ys))
                if fw > 0 and fh > 0:
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
                current_pupil_coords.append(norm_coords)

            # 화면 표시
            display_frame = frame.copy()
            cv2.circle(display_frame, screen_pt, DOT_RADIUS, (0,0,255), -1)
            cv2.putText(display_frame, f"Point {idx+1}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Calibration", display_frame)

            if cv2.waitKey(1) == 27:
                webcam.release()
                cv2.destroyAllWindows()
                return

        if current_pupil_coords:
            median_coords = np.median(current_pupil_coords, axis=0)
            pupil_points.append(tuple(median_coords))
            print(f"[INFO] Point {idx+1} 수집 완료: median 정규화 좌표 {median_coords}")
        else:
            print(f"[WARN] Point {idx+1} 좌표 없음")

    webcam.release()
    cv2.destroyAllWindows()

    if len(pupil_points) == len(screen_points):
        print("[INFO] 시선 매핑 학습 시작...")
        mapper.train(pupil_points, screen_points)
    else:
        print(f"[ERROR] 매핑 학습 실패: {len(pupil_points)}/{len(screen_points)}")

if __name__ == "__main__":
    main()

