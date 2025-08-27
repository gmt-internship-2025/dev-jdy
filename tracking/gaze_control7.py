# gaze_control7.py (Jetson Orin)

import cv2
import pyautogui
import pickle
import os
import numpy as np
import time
from math import hypot

from calibration.gaze_tracking2 import GazeTracking
from calibration.my_calibrator import Calibrator

# 화면 해상도
SCREEN_WIDTH, SCREEN_HEIGHT = 1840, 960

# 웹캠 해상도
CAM_WIDTH = 1840
CAM_HEIGHT = 960  # [FIX]

# ===== 노이즈 억제 파라미터 =====
ema_alpha = 0.35       # EMA 강도
DEADZONE_PX = 25       # 이 픽셀 이내 움직임 무시
SMOOTH_STEP_PX = 40    # 프레임당 최대 이동(픽셀)

# ===== OpenCV 런타임 최적화 =====
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(2, os.cpu_count() // 2))
except Exception:
    pass

# ===== GStreamer 파이프라인 =====
def build_gst_pipelines(w, h, fps=30, device_index=0, flip_lr=True):
    flip_opt = "flip-method=2" if flip_lr else ""
    gst_csi = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv {flip_opt} ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    ).replace("  ", " ").strip()
    gst_usb = (
        f"v4l2src device=/dev/video{device_index} ! "
        f"image/jpeg, width=(int){w}, height=(int){h}, framerate=(fraction){fps}/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw, format=(string)BGR ! "
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
    cap = cv2.VideoCapture(0)
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
if os.path.exists("calibration_data_1.pkl"):
    with open("calibration_data_1.pkl", "rb") as f:
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

        print(f"[+] Loaded {len(X)} samples and fitted scaler & models.")
else:
    print("[!] calibration_data_1.pkl 파일이 없습니다.")
    raise SystemExit(1)

# ===== 카메라 시작 =====
webcam, backend = try_open_camera(CAM_WIDTH, CAM_HEIGHT, fps=30, device_index=0)
if webcam is None:
    print("[ERROR] 카메라 열기 실패")
    raise SystemExit(1)

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
webcam.set(cv2.CAP_PROP_FPS, 30)
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

pyautogui.FAILSAFE = False
frame_count = 0
last_log_time = time.time()

def clamp_to_screen(pt):
    x = max(0, min(SCREEN_WIDTH - 1, int(pt[0])))
    y = max(0, min(SCREEN_HEIGHT - 1, int(pt[1])))
    return (x, y)

while True:
    ok, frame = webcam.read()
    if not ok:
        continue

    # CSI 경로는 HW flip 완료, 나머지에만 flip
    if "V4L2" in backend or "USB(GStreamer" in backend:
        frame = cv2.flip(frame, 1)

    gaze.refresh(frame)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    if left_pupil and right_pupil:
        lp = np.array(left_pupil, dtype=np.float32)
        rp = np.array(right_pupil, dtype=np.float32)
        pupil_avg = ((lp + rp) * 0.5)

        inv_w = 1.0 / CAM_WIDTH
        inv_h = 1.0 / CAM_HEIGHT
        pupil_norm = (pupil_avg[0] * inv_w, pupil_avg[1] * inv_h)

        amp = 15.0
        pupil_amp = ((pupil_norm[0] - 0.5) * amp, (pupil_norm[1] - 0.5) * amp)

        pred_norm = calibrator.predict(pupil_amp)
        pred_norm = np.clip(pred_norm, 0.0, 1.0)

        # --- EMA ---
        if ema_pred is None:
            ema_pred = pred_norm.astype(np.float32)
        else:
            ema_pred = (1.0 - ema_alpha) * ema_pred + ema_alpha * pred_norm

        # 원하는 위치(EMA 적용값)를 픽셀로
        desired_screen = (ema_pred[0] * SCREEN_WIDTH, ema_pred[1] * SCREEN_HEIGHT)

        # --- Dead Zone + Slew Limit ---
        if last_stable_screen is None:
            last_stable_screen = clamp_to_screen(desired_screen)
        else:
            dx = desired_screen[0] - last_stable_screen[0]
            dy = desired_screen[1] - last_stable_screen[1]
            dist = hypot(dx, dy)

            if dist < DEADZONE_PX:
                # 데드존: 출력 갱신하지 않음
                pass
            else:
                # 프레임당 최대 이동량 제한 (슬루)
                if dist > SMOOTH_STEP_PX:
                    scale = SMOOTH_STEP_PX / dist
                    nx = last_stable_screen[0] + dx * scale
                    ny = last_stable_screen[1] + dy * scale
                else:
                    nx = last_stable_screen[0] + dx
                    ny = last_stable_screen[1] + dy
                last_stable_screen = clamp_to_screen((nx, ny))

        # 마우스 이동 (원하면 주석 해제)
        # pyautogui.moveTo(last_stable_screen[0], last_stable_screen[1], duration=0.05)

        # 디버그
        frame_count += 1
        now = time.time()
        if frame_count % 10 == 0 and (now - last_log_time) > 0.3:
            print(f"pred_norm:{tuple(pred_norm)} ema:{tuple(ema_pred)} "
                  f"desired:{(int(desired_screen[0]), int(desired_screen[1]))} "
                  f"stable:{last_stable_screen}")
            last_log_time = now

    annotated = gaze.annotated_frame()
    if last_stable_screen is not None:
        cv2.putText(annotated, f"Stable: {last_stable_screen}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        cv2.circle(annotated, last_stable_screen, 15, (255, 0, 0), 2)

    cv2.imshow("Gaze Mouse Control", annotated)
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

