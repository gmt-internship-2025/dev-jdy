# analyze_perf.py
import subprocess
import re
import matplotlib.pyplot as plt

def run_tegrastats(log_file="perf_log.txt"):
    # tegrastats 실행 → 로그 파일 저장
    return subprocess.Popen(
        ["tegrastats", "--interval", "1000"], 
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT
    )

def run_ffmpeg(log_file="fps_log.txt"):
    # ffmpeg로 웹캠 입력 FPS 추정 → 로그 파일 저장
    return subprocess.Popen(
        ["ffmpeg", "-f", "v4l2", "-i", "/dev/video0", "-an", "-sn", "-dn", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=open(log_file, "w")
    )

def parse_tegrastats(log_file="perf_log.txt"):
    gpu_usage, cpu_usage = [], []
    with open(log_file) as f:
        for line in f:
            g = re.search(r"GR3D_FREQ (\d+)%", line)   # GPU 사용률
            c = re.findall(r"\d+%@", line)             # CPU per core
            if g and c:
                gpu_usage.append(int(g.group(1)))
                cpu_vals = [int(x[:-2]) for x in c]
                cpu_usage.append(sum(cpu_vals)/len(cpu_vals))
    return gpu_usage, cpu_usage

def parse_ffmpeg(log_file="fps_log.txt"):
    fps_vals = []
    with open(log_file) as f:
        for line in f:
            m = re.search(r"fps=\s*([\d\.]+)", line)
            if m:
                fps_vals.append(float(m.group(1)))
    return fps_vals

if __name__ == "__main__":
    print("[INFO] 성능 로그 수집 시작...")
    ts_proc = run_tegrastats()
    ff_proc = run_ffmpeg()

    try:
        input("[ENTER] 키를 누르면 수집을 중단하고 그래프를 그립니다...\n")
    finally:
        ts_proc.terminate()
        ff_proc.terminate()

    print("[INFO] 로그 파싱 및 그래프 생성...")
    gpu_usage, cpu_usage = parse_tegrastats()
    fps_vals = parse_ffmpeg()

    plt.figure()
    if fps_vals: plt.plot(fps_vals, label="FPS")
    if gpu_usage: plt.plot(gpu_usage, label="GPU %")
    if cpu_usage: plt.plot(cpu_usage, label="CPU %")
    plt.legend(); plt.title("Jetson Orin 성능 분석")
    plt.show()

