### GazeTracking 눈 crop 알고리즘 수정

~/…/GazeTracking/gaze_tracking의 eye.py 코드 수정

### 1. margin 값 확장

- 눈을 중심으로 얼마나 넓게 주변을 포함해서 잘라낼 것인지를 의미함.
- 눈을 분석하기 위해서 얼굴 이미지에서 눈 부분을 잘라내는데 너무 딱 맞게 자르면 동공이나 눈썹, 주변 정보가 잘릴 수 있음.

→ margin을 크게 하면 눈동자와 주변 영역 정보를 더 포함할 수 있어서 추적 알고리즘의 오차도 줄어들 가능성이 있음.

- 기존 코드 (margin = 5)

```python
# Cropping on the eye
margin = 5
min_x = np.min(region[:, 0]) - margin
max_x = np.max(region[:, 0]) + margin
min_y = np.min(region[:, 1]) - margin
max_y = np.max(region[:, 1]) + margin
```

- 수정 코드 (margin = 15)

```python
# 눈 영역 크롭 여유 범위 늘리기
margin = 15

height, width = frame.shape[:2]
min_x = max(0, np.min(region[:, 0]) - margin)
max_x = min(width, np.max(region[:, 0]) + margin)
min_y = max(0, np.min(region[:, 1]) - margin)
max_y = min(height, np.max(region[:, 1]) + margin)
```

### 2. 마스킹 하지 않고 단순히 박스로 눈을 크롭하기

- 기존 코드

```python
# Applying a mask to get only the eye
height, width = frame.shape[:2]
black_frame = np.zeros((height, width), np.uint8)
mask = np.full((height, width), 255, np.uint8)
cv2.fillPoly(mask, [region], (0, 0, 0))
eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)
```

```python
...
cv2.fillPoly(mask, [region], (0, 0, 0))  # 눈 영역 마스크 생성
eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)  # 눈 이외는 제거
```

- 얼굴 전체에서 눈 부분만 "폴리곤"으로 따서 마스크(mask) 처리한 후 자름
- 눈 바깥 부분은 모두 검은색(0) 으로 처리해버림

- 단순 박스 크롭 방식

```python
self.frame = frame[min_y:max_y, min_x:max_x]
```

- 마스크 없이 그냥 네모 영역으로 잘라냄
- 눈이 포함된 사각형 박스를 그대로 따는 방식
- 동공, 눈썹, 위아래 여백까지 통째로 포함 → 정보 손실 적음
