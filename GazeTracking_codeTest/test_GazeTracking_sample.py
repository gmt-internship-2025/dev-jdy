import sys
import cv2

sys.path.append("../GazeTracking")

from gaze_tracking import GazeTracking

# GazeTracking 객체 생성
gaze = GazeTracking()

# 기본 웹캠 열기 (장치 번호 0번)
webcam = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    _, frame = webcam.read()

    # 프레임 분석
    gaze.refresh(frame)

    # 분석된 프레임 가져오기
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    # 화면에 상태 출력
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 0, 0), 2)

    # 영상 출력
    cv2.imshow("Gaze Tracking", frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 리소스 정리
webcam.release()
cv2.destroyAllWindows()

