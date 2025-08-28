# conf.py

# --- 해상도 설정 ---
# [MODIFIED] 키오스크 API의 세로 해상도에 맞게 수정
# LOGICAL_WIDTH = 1080
# LOGICAL_HEIGHT = 1920
LOGICAL_WIDTH = 540
LOGICAL_HEIGHT = 960


# --- 카메라 설정 ---
# [MODIFIED] 실제 웹캠에서 사용할 해상도. 일반적인 웹캠 비율(가로)로 설정하여 안정적인 입력값을 받습니다.
# CAM_WIDTH = 1280
# CAM_HEIGHT = 720
CAM_WIDTH = 540
CAM_HEIGHT = 960
CAM_FPS = 30 

# --- 화면 설정 ---
# API 모드에서는 직접 사용되지 않지만, 일관성을 위해 업데이트합니다.
SCREEN_WIDTH = LOGICAL_WIDTH
SCREEN_HEIGHT = LOGICAL_HEIGHT

# --- 시선 좌표 스무딩 (민감도) 설정 ---
EMA_ALPHA = 0.35
DEADZONE_PX = 25
SMOOTH_STEP_PX = 40

# --- 정확도 및 성능 튜닝 파라미터 ---
PUPIL_AMP_FACTOR = 15.0
RIDGE_ALPHA = 0.5
FACE_ROI_SCALE = 2.0

# --- 캘리브레이션 설정 ---
CALIBRATION_FILE = "calibration_data_1.pkl"
POINT_DISPLAY_TIME = 1.0  # 이 값은 calibration_ui6.py 실행 시에만 사용됩니다.

# --- 서버 설정 ---
# API 모드에서는 사용되지 않습니다.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
POST_URL = f"http://localhost:{SERVER_PORT}/api/gaze"

