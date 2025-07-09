import dlib
import cv2

# Dlib의 얼굴 인식 모델 로드
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 이미지 로드
image = cv2.imread("test.jpg")

# 그레이스케일 이미지로 변환 (얼굴 검출을 위한 필수 단계)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = detector(gray)

# 각 얼굴에 대해 랜드마크 추출
for face in faces:
    landmarks = predictor(gray, face)
    
    # 모든 랜드마크 점을 이미지에 표시
    for i, point in enumerate(landmarks.parts()):
        x, y = point.x, point.y
        
        # 각 점에 빨간색 원 그리기
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        
        # 점의 인덱스를 이미지에 표시
        cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

# 결과 이미지 표시
cv2.imshow("Landmarks", image)

# 'q' 키를 누르면 종료
cv2.waitKey(0)
cv2.destroyAllWindows()

