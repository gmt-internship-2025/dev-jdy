import dlib
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# dlib의 얼굴 인식 모델 로드
detector = dlib.get_frontal_face_detector()

# dlib의 얼굴 랜드마크 모델 로드 (다운로드한 .dat 파일 경로 지정)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 이미지 로드
image = io.imread("test.jpg")

# 이미지 크기 조정 (500x500으로 조정)
image_resized = resize(image, (500, 500), anti_aliasing=True)

# 이미지가 BGR일 경우 RGB로 변환 (OpenCV에서 이미지가 BGR로 읽히기 때문)
if image_resized.ndim == 3 and image_resized.shape[2] == 3:
    image_resized = image_resized[:, :, ::-1]  # BGR -> RGB 변환

# 이미지를 0-255 범위로 확장하고, uint8로 변환
image_resized = (image_resized * 255).astype(np.uint8)

# 얼굴 검출
faces = detector(image_resized)

# 이미지에 얼굴을 표시
fig, ax = plt.subplots()
ax.imshow(image_resized)

# 검출된 얼굴에 대해 박스 표시 및 랜드마크 추출
for i, face in enumerate(faces):
    print(f"Face {i}: Left: {face.left()}, Top: {face.top()}, Right: {face.right()}, Bottom: {face.bottom()}")
    
    # 얼굴 영역을 빨간색 사각형으로 표시
    rect = patches.Rectangle((face.left(), face.top()), face.width(), face.height(), linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # 얼굴 랜드마크 추출
    landmarks = predictor(image_resized, face)

    # 왼쪽 눈과 오른쪽 눈의 좌표 추출
    left_eye = landmarks.parts()[36:42]  # 왼쪽 눈 (6개 점)
    right_eye = landmarks.parts()[42:48]  # 오른쪽 눈 (6개 점)
    
    # 왼쪽 눈의 중심 계산
    left_eye_center = np.mean([(point.x, point.y) for point in left_eye], axis=0)
    right_eye_center = np.mean([(point.x, point.y) for point in right_eye], axis=0)
    
    # 눈 중앙을 빨간색 원으로 표시
    ax.add_patch(patches.Circle((left_eye_center[0], left_eye_center[1]), 2, color='r'))
    ax.add_patch(patches.Circle((right_eye_center[0], right_eye_center[1]), 2, color='r'))

    # 눈동자 위치 출력 (좌표)
    print(f"Left eye center: {left_eye_center}")
    print(f"Right eye center: {right_eye_center}")

# 결과 표시
plt.show()
