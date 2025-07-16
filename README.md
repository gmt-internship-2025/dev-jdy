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
