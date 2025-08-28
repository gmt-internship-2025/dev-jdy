# conf.py

# --- 해상도 설정 ---
LOGICAL_WIDTH = 1840
LOGICAL_HEIGHT = 960

# --- 카메라 설정 ---
CAM_WIDTH = LOGICAL_WIDTH
CAM_HEIGHT = LOGICAL_HEIGHT
CAM_FPS = 30

# --- 화면 설정 ---
SCREEN_WIDTH = LOGICAL_WIDTH
SCREEN_HEIGHT = LOGICAL_HEIGHT

# --- 시선 좌표 스무딩 (민감도) 설정 ---
EMA_ALPHA = 0.35
DEADZONE_PX = 25
SMOOTH_STEP_PX = 40

# --- [NEW] 정확도 및 성능 튜닝 파라미터 ---
# 동공 좌표의 미세한 움직임을 증폭시키는 계수입니다.
# 값이 크면 포인터가 더 넓게 움직이지만, 떨림에 민감해질 수 있습니다. (10.0 ~ 20.0 사이에서 조절)
PUPIL_AMP_FACTOR = 15.0

# 캘리브레이션에 사용되는 Ridge 회귀 모델의 정규화 강도(alpha)입니다.
# 값이 크면 모델이 단순해져 떨림이 줄지만, 특정 지점에 대한 정확도는 낮아질 수 있습니다. (0.1 ~ 1.0 사이에서 조절)
RIDGE_ALPHA = 0.5

# 얼굴 추적을 놓쳤을 때, 마지막 위치 주변으로 얼마나 넓게 탐색할지 결정하는 배율입니다.
# 값이 크면 빠르게 움직이는 얼굴을 다시 찾을 확률이 높지만, 탐색 속도는 약간 느려집니다. (1.5 ~ 2.5 사이에서 조절)
FACE_ROI_SCALE = 2.0


# --- 캘리브레이션 설정 ---
CALIBRATION_FILE = "calibration_data_1.pkl"
POINT_DISPLAY_TIME = 1.0

# --- 서버 설정 ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
POST_URL = f"http://localhost:{SERVER_PORT}/api/gaze"

