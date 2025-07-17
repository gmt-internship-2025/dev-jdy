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
