### 성능분석 코드 실행 방법

1. 루트 디렉토리 이동
    
    ```bash
    cd ~/Desktop/GMT/myGazeTracking2_cpu
    ```
    
2. 터미널 A에서 GPU 모니터 시작
    
    ```bash
    python3 -m tools.perf_monitor
    ```
    
    → perf_tegrastats.csv 생성됨 (실행 중 유지)
    
3. 터미널 B에서 FPS/지연 측정 시작
    
    ```bash
    python3 -m tools.bench_gaze \
      --calib calibration_data_1.pkl \
      --out-csv tools/logs/perf_frames.csv \
      --camera 0 --show
    ```
    
    - ESC 또는 Ctrl+C로 종료
    - 결과: tools/logs/perf_frames.csv
    
4. 둘 다 끝난 후 요약/그래프 생성
    
    ```bash
    python3 -m tools.summarize_perf \
      --frames tools/logs/perf_frames.csv \
      --tegra perf_tegrastats.csv \
      --outdir tools/logs/summary
    ```
    
    - 결과물:
        - summary_metrics.csv (평균/중앙값/95p 요약)
        - plot_fps_gpu.png (FPS vs GPU%)
        - hist_t_total_ms.png 등 히스토그램
        - box_fps_per_sec.png (초 단위 FPS 분포)

<aside>

### CPU

```bash
=== PERF SUMMARY ===
              count  mean    median  p95     min     max
t_total_ms    151.0  327.84  328.17  338.10  316.21  374.71
t_refresh_ms  151.0  327.12  327.29  337.22  316.14  373.77
t_predict_ms  151.0    0.72    0.87    0.92    0.07    1.02
fps_inst      151.0    2.90    2.92    3.04    0.58    3.05
fps_avg       151.0    2.81    2.93    2.96    0.58    2.96
```

</aside>

<img width="526" height="390" alt="cpu" src="https://github.com/user-attachments/assets/908cedc5-8aa9-46b0-89d0-f2decaa3f92e" />
