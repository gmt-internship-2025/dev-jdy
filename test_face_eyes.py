import mediapipe as mp
import cv2
import numpy as np

# Mediapipe 얼굴 감지 및 랜드마크 모델 로드
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 얼굴 감지 모델 로드
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# 얼굴 랜드마크 모델 로드
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# 이미지 파일 로드
image = cv2.imread("test.jpg")

# 이미지를 BGR에서 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 얼굴 감지
results = face_detection.process(image_rgb)

# 얼굴이 감지되면
if results.detections:
    for detection in results.detections:
        # 얼굴에 사각형 그리기
        mp_drawing.draw_detection(image, detection)

        # 얼굴 좌표값 출력
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        print(f"Face Bounding Box: x: {x}, y: {y}, w: {w}, h: {h}")

# 얼굴 랜드마크 추적
results_mesh = face_mesh.process(image_rgb)

if results_mesh.multi_face_landmarks:
    for face_landmarks in results_mesh.multi_face_landmarks:
        # 왼쪽 눈과 오른쪽 눈의 랜드마크만 추출
        # 왼쪽 눈: 랜드마크 33번 (눈의 중심)
        left_eye = face_landmarks.landmark[33]
        # 오른쪽 눈: 랜드마크 263번 (눈의 중심)
        right_eye = face_landmarks.landmark[263]

        # 좌표 출력 (정규화된 값)
        print(f"Left Eye: ({left_eye.x}, {left_eye.y})")
        print(f"Right Eye: ({right_eye.x}, {right_eye.y})")

        # 눈에 빨간색 원을 표시
        height, width, _ = image.shape
        left_eye_pos = (int(left_eye.x * width), int(left_eye.y * height))
        right_eye_pos = (int(right_eye.x * width), int(right_eye.y * height))

        # 왼쪽 눈과 오른쪽 눈에만 점을 찍음
        cv2.circle(image, left_eye_pos, 5, (0, 0, 255), -1)  # 왼쪽 눈
        cv2.circle(image, right_eye_pos, 5, (0, 0, 255), -1)  # 오른쪽 눈

# 결과 이미지 표시
cv2.imshow("Face and Eye Detection", image)

# 'q' 키를 누르면 종료
cv2.waitKey(0)
cv2.destroyAllWindows()

