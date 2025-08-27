### 성능분석 코드 실행 방법

1. 루트 디렉토리 이동
    
    ```bash
    cd ~/Desktop/GMT/myGazeTracking2_gpu
    ```
    
2. 터미널 A에서 GPU 모니터 시작
    
    ```bash
    python3 -m tools.perf_monitor_gpu
    ```
    
    → perf_tegrastats.csv 생성됨 (실행 중 유지)
    
3. 터미널 B에서 FPS/지연 측정 시작
    
    ```bash
    python3 -m tools.bench_gaze_gpu \
      --calib calibration_data_1.pkl \
      -out-csv tools/logs/perf_frames_gpu.csv \
      --camera 0 --show
    ```
    
    - ESC 또는 Ctrl+C로 종료
    - 결과: tools/logs/perf_frames.csv
    
4. 둘 다 끝난 후 요약/그래프 생성
    
    ```bash
    python3 -m tools.summarize_perf_gpu \
      --frames tools/logs/perf_frames_gpu.csv \
      --tegra perf_tegrastats.csv \
      --outdir tools/logs/summary_gpu
    ```
    
    - 결과물:
        - summary_metrics.csv (평균/중앙값/95p 요약)
        - plot_fps_gpu.png (FPS vs GPU%)
        - hist_t_total_ms.png 등 히스토그램
        - box_fps_per_sec.png (초 단위 FPS 분포)

### GPU

```bash
=== PERF SUMMARY ===
              count  mean    median p95    min    max
t_total_ms    349.0  64.04   64.09  71.28  45.97  83.83
t_refresh_ms  349.0  62.78   62.80  69.69  45.93  82.16
t_predict_ms  349.0   1.26    1.24   1.65   0.04   2.57
fps_inst      349.0   5.00    5.00   5.32   0.70   9.14
fps_avg       349.0   4.97    5.00   5.01   0.70   5.07
```

<img width="526" height="390" alt="gpu" src="https://github.com/user-attachments/assets/089d8216-c126-430c-8324-4a6343cdb41b" />
