# analyze_accuracy.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 불러오기
df = pd.read_csv("accuracy_log.csv", header=None,
                 names=["pupil_norm_x","pupil_norm_y",
                        "pupil_amp_x","pupil_amp_y",
                        "pred_norm_x","pred_norm_y",
                        "pred_screen_x","pred_screen_y"])

# 오차 계산 (정규화 좌표 기준)
errors = np.sqrt((df["pred_norm_x"]-df["pupil_norm_x"])**2 + 
                 (df["pred_norm_y"]-df["pupil_norm_y"])**2)

print(f"[INFO] Mean Error: {errors.mean():.4f}")
print(f"[INFO] RMSE: {np.sqrt(np.mean(errors**2)):.4f}")

# 오차 분포
plt.hist(errors, bins=30, alpha=0.7)
plt.title("Prediction Error Distribution")
plt.xlabel("Error (normalized units)")
plt.ylabel("Count")
plt.show()

# 산점도 (실제 vs 예측)
plt.scatter(df["pupil_norm_x"], df["pupil_norm_y"], label="Input (pupil norm)", alpha=0.5)
plt.scatter(df["pred_norm_x"], df["pred_norm_y"], label="Predicted (screen norm)", alpha=0.5)
plt.legend(); plt.title("Prediction vs Input")
plt.show()

