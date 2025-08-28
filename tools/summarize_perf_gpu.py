# tools/summarize_perf.py
import os, argparse, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def p95(s: pd.Series):
    return np.percentile(s.dropna(), 95) if len(s.dropna()) else np.nan

def load_frames(path):
    df = pd.read_csv(path)
    # ts(문자열) → 초 단위 버킷(1s)로 맞춤
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    df['sec'] = df['ts'].dt.floor('S')
    return df

def load_tegrastats(path):
    df = pd.read_csv(path)
    # 컬럼: ts,gpu_pct,emc_pct,ram_used_mb,ram_total_mb,cpu_raw
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    df['sec'] = df['ts'].dt.floor('S')
    return df

def summarize(df_frames, df_tegra, outdir):
    os.makedirs(outdir, exist_ok=True)

    # --- 프레임 지연/FPS 요약 ---
    metrics = {}
    for col in ['t_total_ms','t_refresh_ms','t_predict_ms','fps_inst','fps_avg']:
        if col in df_frames.columns:
            s = df_frames[col]
            metrics[col] = {
                'count': int(s.dropna().size),
                'mean': float(s.dropna().mean()) if s.dropna().size else np.nan,
                'median': float(s.dropna().median()) if s.dropna().size else np.nan,
                'p95': float(p95(s)) if s.dropna().size else np.nan,
                'min': float(s.dropna().min()) if s.dropna().size else np.nan,
                'max': float(s.dropna().max()) if s.dropna().size else np.nan,
            }

    df_metrics = pd.DataFrame(metrics).T
    df_metrics.to_csv(os.path.join(outdir, "summary_metrics.csv"))

    # --- 1초 버킷으로 FPS/GPU 정렬 후 조인 ---
    fps_by_sec = df_frames.groupby('sec')['fps_avg'].mean().rename('fps_sec')
    gpu_by_sec = df_tegra.groupby('sec')['gpu_pct'].mean().rename('gpu_pct')
    emc_by_sec = df_tegra.groupby('sec')['emc_pct'].mean().rename('emc_pct')
    ram_by_sec = df_tegra.groupby('sec')['ram_used_mb'].mean().rename('ram_used_mb')

    join = pd.concat([fps_by_sec, gpu_by_sec, emc_by_sec, ram_by_sec], axis=1)
    join.to_csv(os.path.join(outdir, "timeline_sec.csv"))

    # --- 그래프 1: 시간에 따른 FPS/ GPU% ---
    plt.figure()
    if 'fps_sec' in join: join['fps_sec'].plot(label='FPS (avg/sec)')
    if 'gpu_pct' in join: join['gpu_pct'].plot(secondary_y=True, label='GPU %')
    plt.title("FPS vs GPU% over time (1s buckets)")
    plt.xlabel("time (sec)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_fps_gpu.png"))
    plt.close()

    # --- 그래프 2: 지연 분포(히스토그램) ---
    for col in ['t_total_ms','t_refresh_ms','t_predict_ms']:
        if col in df_frames.columns and df_frames[col].dropna().size:
            plt.figure()
            df_frames[col].dropna().plot(kind='hist', bins=40)
            plt.title(f"Latency distribution: {col}")
            plt.xlabel("ms")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"hist_{col}.png"))
            plt.close()

    # --- 그래프 3: 구간별 FPS 박스플롯 (초 단위) ---
    if 'sec' in df_frames.columns and 'fps_avg' in df_frames.columns:
        g = df_frames.groupby('sec')['fps_avg'].apply(list)
        secs = list(range(len(g)))
        data = g.tolist()
        plt.figure()
        plt.boxplot(data, showfliers=False)
        plt.title("Per-second FPS distribution (boxplot)")
        plt.xlabel("time bucket (sec)")
        plt.ylabel("fps")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "box_fps_per_sec.png"))
        plt.close()

    # --- 간단 콘솔 리포트 ---
    print("\n=== PERF SUMMARY ===")
    print(df_metrics.round(2))
    print("\nSaved to:", outdir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="logs/perf_frames.csv")
    ap.add_argument("--tegra",  default="perf_tegrastats.csv")
    ap.add_argument("--outdir", default="logs/summary")
    args = ap.parse_args()

    df_frames = load_frames(args.frames)
    df_tegra  = load_tegrastats(args.tegra)
    summarize(df_frames, df_tegra, args.outdir)

if __name__ == "__main__":
    main()

