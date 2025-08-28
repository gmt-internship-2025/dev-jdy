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
import conf # 설정 파일 import

# ===== OpenCV 런타임 최적화 =====
cv2.setUseOptimized(True)
try:
    import os as _os
    cv2.setNumThreads(max(2, _os.cpu_count() // 2))
except Exception:
    pass

# ===== GStreamer 파이프라인 =====
def build_gst_pipelines(w, h, fps=30, device_index=0, flip_lr=True):
    flip_opt = "flip-method=2" if flip_lr else ""
    # CSI 카메라
    gst_csi = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv {flip_opt} ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    ).replace("  ", " ").strip()
    # USB 카메라 (MJPG 권장)
    gst_usb = (
        f"v4l2src device=/dev/video{device_index} ! "
        f"image/jpeg, width=(int){w}, height=(int){h}, framerate=(fraction){fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )
    return gst_csi, gst_usb

def try_open_camera(w, h, fps=30, device_index=0):
    gst_csi, gst_usb = build_gst_pipelines(w, h, fps, device_index, flip_lr=True)
    cap = cv2.VideoCapture(gst_csi, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap, "CSI(GStreamer, flip in HW)"
    cap = cv2.VideoCapture(gst_usb, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap, "USB(GStreamer, MJPG)"
    cap = cv2.VideoCapture(0)  # 폴백: V4L2
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        return cap, "V4L2(Fallback)"
    return None, "OpenFailed"

# ===== 객체 초기화 =====
gaze = GazeTracking()
calibrator = Calibrator()

# === 스케일러 → 변환 → 회귀 재학습 ===
# conf.py의 파일 경로 사용
if os.path.exists(conf.CALIBRATION_FILE):
    with open(conf.CALIBRATION_FILE, "rb") as f:
        data = pickle.load(f)
        calibrator.X = data["X"]
        calibrator.Y_x = data["Y_x"]
        calibrator.Y_y = data["Y_y"]

        X  = np.asarray(calibrator.X,  dtype=np.float32)
        Yx = np.asarray(calibrator.Y_x, dtype=np.float32)
        Yy = np.asarray(calibrator.Y_y, dtype=np.float32)

        Xs = calibrator.scaler_X.fit_transform(X)
        calibrator.reg_x.fit(Xs, Yx)
        calibrator.reg_y.fit(Xs, Yy)
        calibrator.fitted = True
else:
    print(f"{conf.CALIBRATION_FILE} not found.")
    raise SystemExit(1)

# 카메라 시작 (conf.py 설정값 사용)
webcam, backend = try_open_camera(conf.CAM_WIDTH, conf.CAM_HEIGHT, fps=conf.CAM_FPS, device_index=0)
if webcam is None:
    print("[ERROR] Failed to open camera")
    raise SystemExit(1)

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, conf.CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, conf.CAM_HEIGHT)
webcam.set(cv2.CAP_PROP_FPS, conf.CAM_FPS)
try:
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

print(f"[INFO] Camera backend: {backend}")

# 워밍업
for _ in range(10):
    webcam.read()

# ===== 스무딩 상태 =====
ema_pred = None
last_stable_screen = None

# ===== HTTP 전송 최적화 =====
session = requests.Session()
POST_TIMEOUT = 0.08
POST_HZ = 60.0
min_post_interval = 1.0 / POST_HZ
last_post_ts = 0.0

def clamp_to_screen(pt):
    # conf.py의 논리 해상도 사용
    x = max(0, min(conf.LOGICAL_WIDTH - 1, int(pt[0])))
    y = max(0, min(conf.LOGICAL_HEIGHT - 1, int(pt[1])))
    return (x, y)

while True:
    ok, frame = webcam.read()
    if not ok:
        continue

    # CSI는 HW flip 완료, 나머지는 소프트웨어 flip
    if "V4L2" in backend or "USB(GStreamer" in backend:
        frame = cv2.flip(frame, 1)

    gaze.refresh(frame)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    if left_pupil and right_pupil:
        lp = np.array(left_pupil, dtype=np.float32)
        rp = np.array(right_pupil, dtype=np.float32)
        pupil_avg = (lp + rp) * 0.5

        # conf.py의 카메라 해상도 사용
        inv_w = 1.0 / conf.CAM_WIDTH
        inv_h = 1.0 / conf.CAM_HEIGHT
        pupil_norm = (pupil_avg[0] * inv_w, pupil_avg[1] * inv_h)

        amp = 15.0
        pupil_amp = ((pupil_norm[0] - 0.5) * amp, (pupil_norm[1] - 0.5) * amp)

        pred_norm = calibrator.predict(pupil_amp)
        pred_norm = np.clip(pred_norm, 0.0, 1.0)

        # --- EMA (conf.py의 민감도 값 사용) ---
        if ema_pred is None:
            ema_pred = pred_norm.astype(np.float32)
        else:
            ema_pred = (1.0 - conf.EMA_ALPHA) * ema_pred + conf.EMA_ALPHA * pred_norm

        # 원하는 위치(EMA 적용값)를 픽셀로 (conf.py의 논리 해상도 사용)
        desired_screen = (ema_pred[0] * conf.LOGICAL_WIDTH, ema_pred[1] * conf.LOGICAL_HEIGHT)

        # --- Dead Zone + Slew Limit (conf.py의 민감도 값 사용) ---
        if last_stable_screen is None:
            last_stable_screen = clamp_to_screen(desired_screen)
        else:
            dx = desired_screen[0] - last_stable_screen[0]
            dy = desired_screen[1] - last_stable_screen[1]
            dist = hypot(dx, dy)

            if dist < conf.DEADZONE_PX:
                # 데드존: 출력 갱신하지 않음
                pass
            else:
                # 프레임당 최대 이동량 제한 (슬루)
                if dist > conf.SMOOTH_STEP_PX:
                    scale = conf.SMOOTH_STEP_PX / dist
                    nx = last_stable_screen[0] + dx * scale
                    ny = last_stable_screen[1] + dy * scale
                else:
                    nx = last_stable_screen[0] + dx
                    ny = last_stable_screen[1] + dy
                last_stable_screen = clamp_to_screen((nx, ny))

        # 전송 (conf.py의 서버 주소 사용)
        now = time.time()
        if (now - last_post_ts) >= min_post_interval:
            try:
                session.post(conf.POST_URL,
                             json={"x": last_stable_screen[0], "y": last_stable_screen[1]},
                             timeout=POST_TIMEOUT)
            except requests.RequestException:
                pass
            last_post_ts = now

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

