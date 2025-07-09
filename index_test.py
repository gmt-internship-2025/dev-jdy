import mediapipe as mp
import cv2

# Mediapipe 얼굴 랜드마크 모델 로드
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 얼굴 랜드마크 모델 초기화
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# 이미지 파일 로드
image = cv2.imread("test.jpg")

# 이미지를 RGB로 변환 (Mediapipe는 RGB 이미지를 사용)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 얼굴 랜드마크 추적
results = face_mesh.process(image_rgb)

# 랜드마크가 감지되면
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # 모든 랜드마크 점을 표시
        for i, landmark in enumerate(face_landmarks.landmark):
            # 랜드마크의 x, y 좌표 (정규화된 값)
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])

            # 각 점에 빨간색 원 그리기
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            # 점의 인덱스를 이미지에 표시 (글자 크기 줄이기)
            cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

# 결과 이미지 표시
cv2.imshow("Landmarks", image)

# 'q' 키를 누르면 종료
cv2.waitKey(0)
cv2.destroyAllWindows()

