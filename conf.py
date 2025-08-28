# conf.py
# 이 파일에서 모든 설정을 관리합니다.

# --- 해상도 설정 ---
# 웹 UI, 카메라, 시선 추정 모델이 사용할 논리적 해상도입니다.
# 실제 카메라나 화면 해상도와 다를 수 있지만, 일관성을 위해 통일하는 것을 권장합니다.
LOGICAL_WIDTH = 1840
LOGICAL_HEIGHT = 960

# --- 카메라 설정 ---
# 실제 웹캠에서 사용할 해상도와 프레임 설정입니다.
CAM_WIDTH = LOGICAL_WIDTH
CAM_HEIGHT = LOGICAL_HEIGHT
CAM_FPS = 30 # 카메라 초당 프레임

# --- 화면 설정 ---
# gaze_control7.py가 마우스를 직접 제어할 때 사용할 실제 화면 해상도입니다.
SCREEN_WIDTH = LOGICAL_WIDTH
SCREEN_HEIGHT = LOGICAL_HEIGHT

# --- 시선 좌표 스무딩 (민감도) 설정 ---
# 이 값들을 조정하여 포인터의 움직임을 부드럽게 만들 수 있습니다.
EMA_ALPHA = 0.35          # 지수이동평균(EMA) 가중치. 값이 클수록 최신 좌표에 빠르게 반응 (덜 부드러움). (0.2 ~ 0.5 권장)
DEADZONE_PX = 25          # 데드존(Dead Zone). 포인터가 이 픽셀 값 이내에서 움직이면 무시합니다. (떨림 방지)
SMOOTH_STEP_PX = 40       # 슬루 제한(Slew Limit). 프레임당 포인터가 이동할 수 있는 최대 픽셀. 값이 작을수록 움직임이 더 부드러워집니다.

# --- 캘리브레이션 설정 ---
CALIBRATION_FILE = "calibration_data_1.pkl" # 캘리브레이션 데이터 저장 파일명
POINT_DISPLAY_TIME = 1.0  # 캘리브레이션 각 점을 응시하는 시간 (초)

# --- 서버 설정 ---
SERVER_HOST = "0.0.0.0"   # Flask 웹 서버 호스트
SERVER_PORT = 5000        # Flask 웹 서버 포트
# gaze_control_web.py가 시선 좌표를 전송할 웹 주소입니다.
POST_URL = f"http://localhost:{SERVER_PORT}/api/gaze"

