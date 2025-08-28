# calibration_ui6.py

import cv2
import time
import numpy as np
import sys
import pickle

# calibration 폴더 내의 모듈들을 import합니다.
from calibration.gaze_tracking2 import GazeTracking
from calibration.my_calibrator import Calibrator
# 프로젝트 루트의 conf.py를 import합니다.
import conf 

def main():
    """1080x1920 세로 해상도에 맞춘 캘리브레이션 UI 실행 함수"""
    
    gaze = GazeTracking()
    calibrator = Calibrator()

    # 웹캠 열기
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.", file=sys.stderr)
        return

    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, conf.CAM_WIDTH)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, conf.CAM_HEIGHT)
    webcam.set(cv2.CAP_PROP_FPS, conf.CAM_FPS)
    time.sleep(1.0) # 카메라 안정화

    # --- [수정됨] 키오스크 해상도에 맞는 전체 화면 창 생성 ---
    window_name = "Calibration UI"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 25개 점 캘리브레이션 루프
    for i in range(len(calibrator.matrix.points)):
        # --- [수정됨] 키오스크 해상도를 기준으로 점 위치 계산 ---
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
                # 양쪽 눈의 평균 좌표를 사용 (더 안정적)
                pupil_avg = ((left_pupil[0] + right_pupil[0]) * 0.5, (left_pupil[1] + right_pupil[1]) * 0.5)
                collected_pupil.append(pupil_avg)

            # --- [수정됨] 카메라 영상이 아닌, 검은 배경의 UI를 화면에 표시 ---
            dot_screen = np.zeros((conf.LOGICAL_HEIGHT, conf.LOGICAL_WIDTH, 3), dtype=np.uint8)
            cv2.circle(dot_screen, target, 30, (0, 255, 0), -1) # 타겟 점 크기 키움
            cv2.putText(dot_screen, f"녹색 점을 응시하세요 ({i+1}/{len(calibrator.matrix.points)})", 
                        (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            cv2.imshow(window_name, dot_screen)

            if cv2.waitKey(1) == 27: # ESC로 중단
                webcam.release()
                cv2.destroyAllWindows()
                print("[중단] 캘리브레이션이 중단되었습니다.")
                return

        # 수집된 데이터 처리
        if collected_pupil:
            pupil_mean = np.mean(np.array(collected_pupil, dtype=np.float32), axis=0)
            pupil_norm = (pupil_mean[0] / conf.CAM_WIDTH, pupil_mean[1] / conf.CAM_HEIGHT)
            pupil_amp = (
                (pupil_norm[0] - 0.5) * conf.PUPIL_AMP_FACTOR,
                (pupil_norm[1] - 0.5) * conf.PUPIL_AMP_FACTOR
            )
            # --- [수정됨] 스크린 좌표는 스크린(논리) 해상도 기준으로 정규화 ---
            target_norm = (target[0] / conf.LOGICAL_WIDTH, target[1] / conf.LOGICAL_HEIGHT)
            calibrator.add(pupil_amp, target_norm)
            print(f"[진행] {i+1}번째 점 수집 완료.")

        calibrator.movePoint()

    print("\n[성공] 캘리브레이션이 완료되었습니다!")
    webcam.release()
    cv2.destroyAllWindows()

    # 학습 데이터 저장
    with open(conf.CALIBRATION_FILE, "wb") as f:
        pickle.dump({"X": calibrator.X, "Y_x": calibrator.Y_x, "Y_y": calibrator.Y_y}, f)
    print(f"[저장] 캘리브레이션 데이터 저장 완료: {conf.CALIBRATION_FILE}")

if __name__ == "__main__":
    main()

