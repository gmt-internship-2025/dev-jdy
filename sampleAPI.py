# sampleAPI.py (안내 문구 영어로 변경 완료)

import cv2
import pickle
import os
import sys
import numpy as np
import time
from math import hypot

# Pillow 관련 import는 제거되었습니다.

from calibration.gaze_tracking2 import GazeTracking
from calibration.my_calibrator import Calibrator
import conf

def open_camera(w, h, fps):
    """GStreamer를 사용하여 CSI 또는 USB 카메라를 여는 함수"""
    pipeline_csi = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
        f"nvvidconv flip-method=2 ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )
    pipeline_usb = (
        f"v4l2src device=/dev/video0 ! "
        f"image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )
    
    cap_csi = cv2.VideoCapture(pipeline_csi, cv2.CAP_GSTREAMER)
    if cap_csi.isOpened():
        print("[INFO] CSI 카메라(GStreamer)를 사용합니다.")
        return cap_csi

    cap_usb = cv2.VideoCapture(pipeline_usb, cv2.CAP_GSTREAMER)
    if cap_usb.isOpened():
        print("[INFO] USB 카메라(GStreamer)를 사용합니다.")
        return cap_usb
    
    cap_fallback = cv2.VideoCapture(0)
    if cap_fallback.isOpened():
        print("[INFO] 일반 방식(V4L2)으로 카메라를 사용합니다.")
        return cap_fallback

    return None

def run_calibration(webcam, gaze, calibrator):
    """캘리브레이션 UI를 실행하고 데이터를 수집하는 함수"""
    window_name = "Calibration UI"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i in range(len(calibrator.matrix.points)):
        target = calibrator.getCurrentPoint(conf.LOGICAL_WIDTH, conf.LOGICAL_HEIGHT)
        target = tuple(map(int, target))

        collected_pupil = []
        start_time = time.time()
        
        while time.time() - start_time < conf.POINT_DISPLAY_TIME:
            ok, frame = webcam.read()
            if not ok: continue

            frame = cv2.flip(frame, 1)
            gaze.refresh(frame)

            if gaze.pupils_located:
                left_pupil = gaze.pupil_left_coords()
                right_pupil = gaze.pupil_right_coords()
                pupil_avg = ((left_pupil[0] + right_pupil[0]) * 0.5, (left_pupil[1] + right_pupil[1]) * 0.5)
                collected_pupil.append(pupil_avg)

            dot_screen = np.zeros((conf.LOGICAL_HEIGHT, conf.LOGICAL_WIDTH, 3), dtype=np.uint8)
            cv2.circle(dot_screen, target, 30, (0, 255, 0), -1)
            
            # --- [수정] 안내 문구를 한글에서 영어로 변경 ---
            text = f"Please look at the green dot ({i+1}/{len(calibrator.matrix.points)})"
            cv2.putText(dot_screen, text, (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)

            cv2.imshow(window_name, dot_screen)

            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                print("[중단] 캘리브레이션이 중단되었습니다.", file=sys.stderr)
                return False

        if collected_pupil:
            pupil_mean = np.mean(np.array(collected_pupil, dtype=np.float32), axis=0)
            pupil_norm = (pupil_mean[0] / conf.CAM_WIDTH, pupil_mean[1] / conf.CAM_HEIGHT)
            pupil_amp = ((pupil_norm[0] - 0.5) * conf.PUPIL_AMP_FACTOR, (pupil_norm[1] - 0.5) * conf.PUPIL_AMP_FACTOR)
            target_norm = (target[0] / conf.LOGICAL_WIDTH, target[1] / conf.LOGICAL_HEIGHT)
            calibrator.add(pupil_amp, target_norm)

        calibrator.movePoint()

    cv2.destroyAllWindows()
    with open(conf.CALIBRATION_FILE, "wb") as f:
        pickle.dump({"X": calibrator.X, "Y_x": calibrator.Y_x, "Y_y": calibrator.Y_y}, f)
    print(f"[성공] 캘리브레이션 데이터 저장 완료: {conf.CALIBRATION_FILE}")
    return True

def run_gaze_api(webcam, gaze, calibrator):
    """캘리브레이션이 완료된 모델로 시선 추적 API를 실행하는 함수"""
    print('\nModels Loaded', flush=True)

    ema_pred = None
    last_stable_screen = (conf.LOGICAL_WIDTH // 2, conf.LOGICAL_HEIGHT // 2)
    face_lost_since = None
    FACE_LOST_TIMEOUT = 5

    while True:
        ok, frame = webcam.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        gaze.refresh(frame)

        if gaze.pupils_located:
            face_lost_since = None
            
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            
            pupil_avg = ((left_pupil[0] + right_pupil[0]) * 0.5, (left_pupil[1] + right_pupil[1]) * 0.5)
            pupil_norm = (pupil_avg[0] / conf.CAM_WIDTH, pupil_avg[1] / conf.CAM_HEIGHT)
            
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
                
                last_stable_screen = (
                    max(0, min(conf.LOGICAL_WIDTH - 1, int(nx))),
                    max(0, min(conf.LOGICAL_HEIGHT - 1, int(ny)))
                )

            print(f"{last_stable_screen[0]},{last_stable_screen[1]}", flush=True)

        else:
            if face_lost_since is None:
                face_lost_since = time.time()
            elif time.time() - face_lost_since > FACE_LOST_TIMEOUT:
                break

def main():
    """메인 파이프라인: 사용자 선택에 따라 캘리브레이션 또는 시선 추적을 실행"""
    
    gaze = GazeTracking()
    calibrator = Calibrator()
    calib_data_exists = os.path.exists(conf.CALIBRATION_FILE)
    
    choice = ''
    if calib_data_exists:
        while choice not in ['y', 'n']:
            choice = input(">>> Use existing calibration data? (y/n): ").lower()
    
    print("\n[INFO] Initializing camera...")
    webcam = open_camera(conf.CAM_WIDTH, conf.CAM_HEIGHT, conf.CAM_FPS)
    if not webcam:
        print("[ERROR] Could not open camera. Check permissions, device index, or GStreamer settings.", file=sys.stderr)
        return
        
    time.sleep(1.0)

    try:
        if choice == 'n' or not calib_data_exists:
            print("\n[INFO] Starting calibration...")
            time.sleep(1)
            success = run_calibration(webcam, gaze, calibrator)
            if not success:
                return 
        else:
            print(f"\n[INFO] Loading existing data from '{conf.CALIBRATION_FILE}'...")
            with open(conf.CALIBRATION_FILE, "rb") as f:
                data = pickle.load(f)
                calibrator.X = data["X"]
                calibrator.Y_x = data["Y_x"]
                calibrator.Y_y = data["Y_y"]
                calibrator.movePoint()
                calibrator.matrix.iterator = 0
        
        run_gaze_api(webcam, gaze, calibrator)

    except KeyboardInterrupt:
        pass
    finally:
        print('Exit eye mode', flush=True)
        if webcam:
            webcam.release()

if __name__ == "__main__":
    main()

