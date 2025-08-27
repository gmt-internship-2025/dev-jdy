# main.py

import subprocess
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DATA_PATH = os.path.join(BASE_DIR, "calibration_data_1.pkl")

def run_calibration():
    print("[1/3] Calibration 시작...")
    # 모듈 실행 방식 (-m) 사용 → 패키지 경로 문제 방지
    subprocess.run(["python3", "-m", "calibration.calibration_ui6"])

    if not os.path.exists(CALIB_DATA_PATH):
        print("[!] Calibration 데이터가 없습니다. 종료합니다.")
        exit(1)

def run_web_server():
    print("[2/3] 웹 서버 실행...")
    # 모듈 실행 방식 (-m) 사용
    proc = subprocess.Popen(["python3", "-m", "web.app1"])
    time.sleep(3)  # 서버 초기화 대기
    return proc

def run_tracking():
    print("[3/3] 시선 추적 시작...")
    # 모듈 실행 방식 (-m) 사용
    subprocess.run(["python3", "-m", "tracking.gaze_control_web"])

if __name__ == "__main__":
    run_calibration()
    web_proc = run_web_server()

    try:
        run_tracking()
    except KeyboardInterrupt:
        pass
    finally:
        web_proc.terminate()
        print("[완료] 파이프라인 종료")

