### 시선 보정 모델 학습 파이프라인 구축

- ‘눈동자 좌표 → 화면 상 시선 위치’를 매핑할 수 있도록 시선 보정 모델 학습 파이프라인을 구축
- 전체 구조 순서
    
    
    | 단계 | 설명 |
    | --- | --- |
    | 1 | 화면에 9개의 점(3x3 grid)을 순차적으로 띄우기 |
    | 2 | 각 점을 2초 정도 응시하도록 유도 |
    | 3 | 응시 중 pupil 좌표 수집 (gaze.pupil_left_coords()) |
    | 4 | 각 점마다 평균 좌표 저장 |
    | 5 | (pupil 좌표) → (화면 좌표) 쌍으로 회귀 모델 학습 |
    | 6 | 이후 pupil 좌표가 들어오면 화면 좌표 예측 가능 |


- 어떤 GUI 방식으로?
    - OpenCV
- 해상도는 몇으로?
    - 640x480
- 그리드 크기는 몇 점으로?
    - 3x3



### 화면 점 좌표 쌍을 회귀 모델로 학습해서 시선 보정 모델 생성

- 디렉코리 구조

```python
GMT/
├── gaze_tracking/               ← (수정 완료) 기존 GazeTracking 라이브러리 
├── calibration_mapping.py       ← (새로 생성) 시선 보정 및 매핑 회귀 모델 관련 코드
├── test_gaze_calibration.py     ← (새로 생성) 점 순차 출력 및 pupil 좌표 수집 및 학습 실행 코드
```

- scikit-learn 라이브러리 설치

```python
pip3 install scikit-learn
```


- calibration 테스트 실행

```bash
python3 test_gaze_calibration.py
```

```bash
python3 test_gaze_prediction.py
```
---

## 입력 데이터 품질 개선

- 눈동자 좌표 수집 시 이동평균(moving average) 또는 중앙값(median)으로 노이즈 줄이기
- 좌/우 눈 모두 사용해서 좌표 평균을 내면 순간 튀는 값이 줄어듦

- 양안 평균 적용
    - gaze.pupil_left_coords()와 gaze.pupil_right_coords()를 둘 다 읽어서 평균내기
- median 적용
    - 한 점에서 2초 동안 모은 pupil 좌표들에 대해 np.median()을 사용하여 대표값 계산

### test_gaze_calibration.py (데이터 수집)
- 카메라가 켜지자마자 calibration이 실행되어 부정확한 좌표값이 수정되는 문제가 있음
- 따라서 카메라가 켜지고 5초 후에 calibration을 실시하도록 수정

```python
# 5초 카운트다운 화면 출력
    for countdown in range(5, 0, -1):  # 5,4,3,2,1
        ret, frame = webcam.read()
        if not ret or frame is None:
            continue
        # 중앙에 카운트 숫자 표시
        text = str(countdown)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
        # 중앙 좌표 계산
        x = (WINDOW_WIDTH - text_width) // 2
        y = (WINDOW_HEIGHT + text_height) // 2
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1000) == 27:  # 1초 기다림, ESC로 중단 가능
            webcam.release()
            cv2.destroyAllWindows()
            return
```

- 화면 해상도를 늘려 큰 화면으로 조정 → 눈동자 움직임 좌표값 더 확실하게 하여 모델학습에 반영

```python
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
```

- 양안 mean 값 계산하여 모델학습에 반영 → 좌우 눈 좌표를 평균값 계산하여 노이즈 완화

```python
# 양안 mean 계산
    left = gaze.pupil_left_coords()
    right = gaze.pupil_right_coords()
    
    if left is not None and right is not None:
        coords = np.mean([left, right], axis=0)
    elif left is not None:
        coords = left
    elif right is not None:
        coords = right
    else:
        coords = None
```

- median 값 계산하여 모델학습에 반영→ 수집한 좌표들의 중앙값 계산하여 순간 튀는 값 제거

```python
# median 계산
    if current_pupil_coords:
        avg_coords = np.median(current_pupil_coords, axis=0)
        pupil_points.append(tuple(avg_coords))
        print(f"[INFO] Point {idx+1} 수집 완료: median 좌표 {avg_coords}")
    else:
        print(f"[WARN] Point {idx+1} 에 대해 pupil 좌표 없음")
```

- calibration grid 3x3에서 5x5로 적용 → 화면 전체에 더 많은 좌표 샘플을 수집

```python
GRID_ROWS, GRID_COLS = 5, 5
```

- 영상송출 delay로 인한 좌표값 부정확 수집 의심으로 calibration display time 2에서 3으로 적용

```python
DISPLAY_TIME = 3
```

### test_gaze_prediction.py (실시간 예측)
- 좌표값이 음수로 출력되어 화면을 벗어나지 않도록 값 조정

```python
# 화면 범위를 벗어나지 않도록 값 조정
px = max(0, min(WINDOW_WIDTH - 1, px))
py = max(0, min(WINDOW_HEIGHT - 1, py))
```

- 화면 해상도를 늘려 큰 화면으로 조정

```python
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
```

- 양안 mean 계산 로직 적용

```python
# 양안 mean 계산
    left = gaze.pupil_left_coords()
    right = gaze.pupil_right_coords()
    if left is not None and right is not None:
        coords = np.mean([left, right], axis=0)
    elif left is not None:
        coords = left
    elif right is not None:
        coords = right
    else:
        coords = None

    if coords is not None and mapper.is_trained:
        pred_x, pred_y = mapper.predict(coords)
```

### 왜 비선형 회귀 / MLP가 필요한가?

- 현재 calibration_mapping.py 에서는 LinearRegression 으로 pupil에서 시선추적 화면 좌표를 직선으로만 매핑하고 있음
- 실제 시선과 pupil 좌표 관계는 왜곡이 심하고 비선형
- 따라서 더 복잡한 모델로 학습해야 함.

1. MLPRegressor (다층 퍼셉트론)
    - 신경망 기반으로 비선형 관계를 잘 모델링할 수 있음.
    
2. Polynomial Regression (다항 회귀)
    - 간단하면서도 비선형 매핑을 지원하는 방법.
    - 기존의 LinearRegression을 PolynomialFeatures 와 결합하면 됨.

- 먼저 MLPRegressor부터 시도
- scikit-learn에 내장되어 있고, LinearRegression 대신 바꾸기만 하면됨.

### 1. MLPRegressor 라이브러리 설치

- scikit-learn은 이미 설치되었을거긴 한데 다시 확인

```jsx
pip3 install scikit-learn
```

### calibration_mapping.py 수정하기

- 기존 코드

```python
from sklearn.linear_model import LinearRegression

class GazeMapper:
    def __init__(self):
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.is_trained = False
```

- 수정 코드 (MLPRegressor)

```python
from sklearn.neural_network import MLPRegressor

class GazeMapper:
    def __init__(self):
        # MLPRegressor: 은닉층 2개 (64, 64), 활성화 함수 relu, 500번 반복
        self.model_x = MLPRegressor(hidden_layer_sizes=(64, 64),
                                    activation='relu',
                                    solver='adam',
                                    max_iter=500,
                                    random_state=42)
        self.model_y = MLPRegressor(hidden_layer_sizes=(64, 64),
                                    activation='relu',
                                    solver='adam',
                                    max_iter=500,
                                    random_state=42)
        self.is_trained = False
```

- 나머지 train(), predict(), save(), load()는 그대로 사용 가능
- (MLPRegressor도 scikit-learn의 estimator이기 때문에 fit/predict 똑같이 동작)
- test_gaze_calibration.py, test_gaze_prediction.py는 그대로 사용
    - GazeMapper의 내부 모델만 바꾼 것이기 때문

- 테스트 실행

```bash
python3 test_gaze_calibration.py
```

```bash
python3 test_gaze_prediction.py
```

### MLPRegressor 적용의 한계

- 정확도가 여전히 낮음 (오히려 떨어짐)
- 데이터 수는 적은데 복잡한 MLP를 썼기 때문 → overfitting/불안정
- 선형모델보다 성능이 떨어지는 현상 발생

→ 간단하면서도 비선형을 어느 정도 지원하는 다항 회귀 시도

→ MLP는 데이터가 충분할 때 강력한 모델이지만 데이터가 적을 때는 오히려 안 좋아질 수 있음

### 2. Polynomial Regression (다항 회귀) 적용해보기

- 기존 LinearRegression 기반으로 아주 간단히 변경 가능.
- 복잡한 신경망보다 훨씬 안정적일 가능성이 높음.

- 기존 코드

```bash
from sklearn.linear_model import LinearRegression

class GazeMapper:
    def __init__(self):
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.is_trained = False
```

- 수정 코드 (Polynomial Regression)

```bash
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

class GazeMapper:
    def __init__(self):
        # PolynomialFeatures + Ridge로 변경
        degree = 2
        self.model_x = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
        self.model_y = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
        self.is_trained = False
        
    def train(self, pupil_coords, screen_coords):
        X = np.array(pupil_coords)
        y_x = np.array([pt[0] for pt in screen_coords])
        y_y = np.array([pt[1] for pt in screen_coords])

        # Ridge 기반 모델 학습
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        self.is_trained = True
        print("[GazeMapper] Training completed (Polynomial Regression)")
```

### 3. 좌표 정규화

- 현재 pupil 좌표는 카메라 해상도(픽셀) 기준으로 나옴
- 얼굴이 카메라 화면 안에서 약간만 움직여도 pupil 좌표가 크게 변해버림
- 따라서 이를 얼굴 영역의 bounding box 기준으로 정규화하면 얼굴 위치/크기 변화에 덜 민감해짐.

- 정규화 적용 방법
    - GazeTracking에서 pupil 좌표를 얻은 직후 해당 프레임의 얼굴 bounding box 기준으로 (0~1) 사이로 정규화한 값을 학습 및 예측에 사용.

- test_gaze_calibration.py
- pupil 좌표를 bounding box 기준으로 정규화해서 학습 데이터로 사용

```python
# 정규화 함수 추가
def normalize_coords(coords, face_box):
    (fx, fy, fw, fh) = face_box
    if fw == 0 or fh == 0:
        return (0.5, 0.5)  # 기본값을 리턴해서 ZeroDivision 방지
    nx = (coords[0] - fx) / fw
    ny = (coords[1] - fy) / fh
    return (nx, ny)
```

```python
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
```

- test_gaze_prediction.py
- 예측 단계에서도 pupil 좌표를 동일하게 정규화해서 mapper.predict()에 전달

```python
# 정규화 함수 동일
def normalize_coords(coords, face_box):
    (fx, fy, fw, fh) = face_box
    if fw == 0 or fh == 0:  # 안전 처리
        return (0.5, 0.5)
    nx = (coords[0] - fx) / fw
    ny = (coords[1] - fy) / fh
    return (nx, ny)
```

```python
...
    norm_coords = normalize_coords(coords, face_box)
    pred_x, pred_y = mapper.predict(norm_coords)
**...**
```
