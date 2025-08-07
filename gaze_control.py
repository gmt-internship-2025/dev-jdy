# gaze_mouse_control.py

import cv2
import pyautogui
from gaze_tracking2 import GazeTracking
from my_calibrator import Calibrator

# 화면 해상도
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# 객체 초기화
gaze = GazeTracking()
calibrator = Calibrator()

# 웹캠 시작
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    # 동공 좌표
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    if left_pupil and right_pupil:
        pupil_avg = (
            (left_pupil[0] + right_pupil[0]) / 2,
            (left_pupil[1] + right_pupil[1]) / 2
        )

        pred = calibrator.predict(pupil_avg)

        # 유효한 위치일 경우만 마우스 이동
        if 0 <= pred[0] <= SCREEN_WIDTH and 0 <= pred[1] <= SCREEN_HEIGHT:
            pyautogui.moveTo(pred[0], pred[1], duration=0.1)

    # 디버그 화면
    annotated = gaze.annotated_frame()
    if left_pupil and right_pupil:
        cv2.putText(annotated, f"Pupil: {pupil_avg}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Pred: {pred}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    cv2.imshow("Gaze Mouse Control", annotated)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

