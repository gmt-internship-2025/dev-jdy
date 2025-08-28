# gaze_control_web.py (Jetson Orin)

import cv2
import pickle
import os
import numpy as np
import requests
import time
from math import hypot
from calibration.gaze_tracking2 import GazeTracking
from calibration.my_calibrator import Calibrator
import conf

# ===== OpenCV 런타임 최적화 =====
cv2.setUseOptimized(True)
if hasattr(os, 'cpu_count'):
    cv2.setNumThreads(max(1, os.cpu_count() // 2))

# ===== GStreamer 파이프라인 =====
def build_gstreamer_pipeline(w, h, fps, is_csi=True):
    if is_csi:
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
            f"nvvidconv flip-method=2 ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
    else: # USB
        return (
            f"v4l2src device=/dev/video0 ! "
            f"image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

def open_camera(w, h, fps):
    pipeline_csi = build_gstreamer_pipeline(w, h, fps, is_csi=True)
    cap = cv2.VideoCapture(pipeline_csi, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap, "CSI(GStreamer)"

    pipeline_usb = build_gstreamer_pipeline(w, h, fps, is_csi=False)
    cap = cv2.VideoCapture(pipeline_usb, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap, "USB(GStreamer)"
        
    cap = cv2.VideoCapture(0) # Fallback
    if cap.isOpened():
        return cap, "V4L2"
        
    return None, "Failed"

# ===== 모델 및 캘리브레이션 데이터 로드 =====
gaze = GazeTracking()
calibrator = Calibrator()

if os.path.exists(conf.CALIBRATION_FILE):
    with open(conf.CALIBRATION_FILE, "rb") as f:
        data = pickle.load(f)
        calibrator.X = data["X"]
        calibrator.Y_x = data["Y_x"]
        calibrator.Y_y = data["Y_y"]
        # 모델 재학습 (프로그램 시작 시 한 번만)
        calibrator.movePoint() # 데이터를 로드하고 _fit_locked를 호출하는 효과
        calibrator.matrix.iterator = 0 # 이터레이터 리셋
        print(f"[INFO] Loaded and fitted {len(calibrator.X)} calibration samples.")
else:
    print(f"[ERROR] {conf.CALIBRATION_FILE} not found. Please run calibration first.")
    raise SystemExit(1)

# ===== 카메라 시작 =====
webcam, backend = open_camera(conf.CAM_WIDTH, conf.CAM_HEIGHT, conf.CAM_FPS)
if not webcam:
    print("[ERROR] Failed to open camera.")
    raise SystemExit(1)
print(f"[INFO] Camera backend: {backend}")

# ===== 스무딩 및 전송 상태 초기화 =====
ema_pred = None
last_stable_screen = (conf.LOGICAL_WIDTH // 2, conf.LOGICAL_HEIGHT // 2)
session = requests.Session()
min_post_interval = 1.0 / 60.0
last_post_ts = 0.0

def clamp_to_screen(x, y):
    return (
        max(0, min(conf.LOGICAL_WIDTH - 1, int(x))),
        max(0, min(conf.LOGICAL_HEIGHT - 1, int(y)))
    )

print("[INFO] Starting gaze control...")
while True:
    ok, frame = webcam.read()
    if not ok:
        time.sleep(0.01)
        continue

    if "V4L2" in backend:
        frame = cv2.flip(frame, 1)

    gaze.refresh(frame)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    if left_pupil and right_pupil:
        pupil_avg = ((left_pupil[0] + right_pupil[0]) * 0.5, (left_pupil[1] + right_pupil[1]) * 0.5)
        
        pupil_norm = (pupil_avg[0] / conf.CAM_WIDTH, pupil_avg[1] / conf.CAM_HEIGHT)
        
        # [MODIFIED] conf.py의 증폭 계수 사용
        pupil_amp = (
            (pupil_norm[0] - 0.5) * conf.PUPIL_AMP_FACTOR,
            (pupil_norm[1] - 0.5) * conf.PUPIL_AMP_FACTOR
        )

        pred_norm = calibrator.predict(pupil_amp)
        pred_norm = np.clip(pred_norm, 0.0, 1.0)

        if ema_pred is None:
            ema_pred = pred_norm
        else:
            ema_pred = (1.0 - conf.EMA_ALPHA) * ema_pred + conf.EMA_ALPHA * pred_norm

        desired_screen = (ema_pred[0] * conf.LOGICAL_WIDTH, ema_pred[1] * conf.LOGICAL_HEIGHT)
        
        dx = desired_screen[0] - last_stable_screen[0]
        dy = desired_screen[1] - last_stable_screen[1]
        dist = hypot(dx, dy)

        if dist > conf.DEADZONE_PX:
            if dist > conf.SMOOTH_STEP_PX:
                scale = conf.SMOOTH_STEP_PX / dist
                nx = last_stable_screen[0] + dx * scale
                ny = last_stable_screen[1] + dy * scale
            else:
                nx = desired_screen[0]
                ny = desired_screen[1]
            last_stable_screen = clamp_to_screen(nx, ny)

        now = time.time()
        if (now - last_post_ts) >= min_post_interval:
            try:
                session.post(conf.POST_URL,
                             json={"x": last_stable_screen[0], "y": last_stable_screen[1]},
                             timeout=0.08)
                last_post_ts = now
            except requests.RequestException:
                pass

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
print("[INFO] Gaze control stopped.")

