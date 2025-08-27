# tools/bench_gaze.py
import os, time, csv, argparse
import numpy as np
import cv2

# 패키지 경로/이름 그대로 사용
from calibration.gaze_tracking2 import GazeTracking
from calibration.my_calibrator import Calibrator

# === 기본 카메라/화면 해상도 (동일 값 사용 권장) ===
CAM_WIDTH, CAM_HEIGHT = 1840, 960
SCREEN_WIDTH, SCREEN_HEIGHT = 1840, 960

def load_calibration(pkl_path="calibration_data_1.pkl", calibrator=None):
    import pickle
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"{pkl_path} not found. 먼저 calibration_ui6.py로 학습 데이터를 생성하세요.")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    calibrator.X = data["X"]; calibrator.Y_x = data["Y_x"]; calibrator.Y_y = data["Y_y"]
    calibrator.reg_x.fit(calibrator.X, calibrator.Y_x)
    calibrator.reg_y.fit(calibrator.X, calibrator.Y_y)
    calibrator.fitted = True

def main(args):
    # ── 객체 초기화 ───────────────────────────────────────────────
    gaze = GazeTracking()
    calibrator = Calibrator()

    # 캘리브레이션 파라미터 로드 (추적 스크립트와 동일 흐름)
    load_calibration(args.calib, calibrator)

    # 카메라 설정
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # 출력 CSV 준비
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    f = open(args.out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "ts", "frame_idx",
        "t_total_ms", "t_refresh_ms", "t_predict_ms",
        "fps_inst", "fps_avg",
        "pupil_norm_x", "pupil_norm_y",
        "pred_norm_x", "pred_norm_y",
        "pred_screen_x", "pred_screen_y"
    ])

    frame_idx = 0
    t0 = time.perf_counter()
    t_last = t0
    fps_hist = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)

            t_start = time.perf_counter()
            # ── GazeTracking 분석 ──────────────────────────────
            t1 = time.perf_counter()
            gaze.refresh(frame)  # 내부에서 얼굴/랜드마크/눈 분리/임계치/동공 검출 :contentReference[oaicite:8]{index=8}
            t2 = time.perf_counter()

            left_p = gaze.pupil_left_coords()
            right_p = gaze.pupil_right_coords()

            pred_norm = np.array([np.nan, np.nan])
            pred_screen = (np.nan, np.nan)
            pupil_norm = (np.nan, np.nan)

            if left_p and right_p:
                # 평균 좌표 → 정규화 → 증폭 → 회귀 예측(기존과 동일 파이프라인) :contentReference[oaicite:9]{index=9}
                pupil_avg = ((left_p[0] + right_p[0]) / 2, (left_p[1] + right_p[1]) / 2)
                pupil_norm = (pupil_avg[0] / CAM_WIDTH, pupil_avg[1] / CAM_HEIGHT)
                pupil_amp = ((pupil_norm[0] - 0.5) * 15, (pupil_norm[1] - 0.5) * 15)
                pred_norm = calibrator.predict(pupil_amp)
                pred_norm = np.clip(pred_norm, 0, 1)
                pred_screen = (int(pred_norm[0] * SCREEN_WIDTH), int(pred_norm[1] * SCREEN_HEIGHT))

            t3 = time.perf_counter()

            # 타이밍/ FPS 계산
            t_refresh = (t2 - t1) * 1000.0
            t_predict = (t3 - t2) * 1000.0
            t_total = (t3 - t1) * 1000.0

            now = time.perf_counter()
            fps_inst = 1.0 / (now - t_last) if (now - t_last) > 0 else 0.0
            t_last = now
            fps_hist.append(fps_inst)
            fps_avg = sum(fps_hist[-60:]) / min(len(fps_hist), 60)

            # 로그 기록
            w.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                frame_idx,
                round(t_total,2), round(t_refresh,2), round(t_predict,2),
                round(fps_inst,2), round(fps_avg,2),
                round(pupil_norm[0],4) if pupil_norm[0]==pupil_norm[0] else "",
                round(pupil_norm[1],4) if pupil_norm[1]==pupil_norm[1] else "",
                round(float(pred_norm[0]),4) if pred_norm[0]==pred_norm[0] else "",
                round(float(pred_norm[1]),4) if pred_norm[1]==pred_norm[1] else "",
                pred_screen[0] if pred_screen[0]==pred_screen[0] else "",
                pred_screen[1] if pred_screen[1]==pred_screen[1] else "",
            ])

            # 옵션: 디버그 시각화
            if args.show:
                annotated = gaze.annotated_frame()  # 동공 십자선 오버레이 :contentReference[oaicite:10]{index=10}
                if left_p and right_p:
                    cv2.putText(annotated, f"FPS:{fps_avg:.1f} T:{t_total:.1f}ms", (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(annotated, f"Pred:{pred_screen}", (20,70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,0), 2)
                    cv2.circle(annotated, pred_screen, 12, (255,0,0), 2)
                cv2.imshow("Bench Gaze", annotated)
                if cv2.waitKey(1) == 27:
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

    finally:
        f.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calibration_data_1.pkl", help="캘리브레이션 pkl 경로")
    ap.add_argument("--camera", type=int, default=0, help="비디오 장치 인덱스")
    ap.add_argument("--out-csv", default="perf_frames.csv", help="프레임 성능 로그 CSV")
    ap.add_argument("--show", action="store_true", help="디버그 화면 표시")
    ap.add_argument("--max-frames", type=int, default=0, help="0=제한 없음")
    args = ap.parse_args()
    main(args)

