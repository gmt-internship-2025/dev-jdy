# my_calibrator.py (Jetson Orin)

import numpy as np
import threading
import conf  # [MODIFIED] 설정 파일 import

_CUML_AVAILABLE = False
try:
    from cuml.linear_model import Ridge as _cuRidge
    from cuml.preprocessing import StandardScaler as _cuScaler
    _CUML_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import Ridge as _skRidge
    from sklearn.preprocessing import StandardScaler as _skScaler

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32))

class CalibrationMatrix:
    def __init__(self):
        self.iterator = 0
        xs = np.linspace(0.0, 1.0, 5)
        ys = np.linspace(0.0, 1.0, 5)
        # [MODIFIED] 지그재그 스캔 패턴으로 변경하여 시선 이동을 자연스럽게 유도
        points = []
        for i, y in enumerate(ys):
            row = xs if i % 2 == 0 else xs[::-1]
            for x in row:
                points.append([x, y])
        self.points = np.array(points, dtype=np.float32)

    def getCurrentPoint(self, width=1.0, height=1.0):
        p = self.points[self.iterator]
        return np.array([p[0] * width, p[1] * height], dtype=np.float32)

    def movePoint(self):
        self.iterator = (self.iterator + 1) % len(self.points)

class Calibrator:
    MIN_SAMPLES_PER_POINT = 30

    def __init__(self):
        self.X, self.Y_x, self.Y_y = [], [], []
        self._tmp_X, self._tmp_Y_x, self._tmp_Y_y = [], [], []

        # [MODIFIED] conf.py에서 alpha 값 가져오기
        ridge_alpha = conf.RIDGE_ALPHA
        if _CUML_AVAILABLE:
            self.reg_x = _cuRidge(alpha=ridge_alpha)
            self.reg_y = _cuRidge(alpha=ridge_alpha)
            self.scaler_X = _cuScaler(with_mean=True, with_std=True)
            self._backend = "GPU(cuML)"
        else:
            self.reg_x = _skRidge(alpha=ridge_alpha)
            self.reg_y = _skRidge(alpha=ridge_alpha)
            self.scaler_X = _skScaler(with_mean=True, with_std=True)
            self._backend = "CPU(sklearn)"

        self.fitted = False
        self.matrix = CalibrationMatrix()
        self.lock = threading.Lock()

    def add(self, pupil_coords, screen_coords):
        with self.lock:
            self._tmp_X.append(np.asarray(pupil_coords, dtype=np.float32).flatten())
            self._tmp_Y_x.append(np.float32(screen_coords[0]))
            self._tmp_Y_y.append(np.float32(screen_coords[1]))
            # [MODIFIED] 첫 학습 트리거 로직 간소화
            if not self.fitted and len(self._tmp_X) >= self.MIN_SAMPLES_PER_POINT:
                self._fit_locked(tmp_only=True)

    def _fit_locked(self, tmp_only=False):
        if tmp_only:
            X_fit, Yx_fit, Yy_fit = self._tmp_X, self._tmp_Y_x, self._tmp_Y_y
        else:
            X_fit = self.X + self._tmp_X
            Yx_fit = self.Y_x + self._tmp_Y_x
            Yy_fit = self.Y_y + self._tmp_Y_y

        if len(X_fit) < 2:  # 학습에 필요한 최소 샘플 수
            return

        X_total = np.asarray(X_fit, dtype=np.float32)
        Yx = np.asarray(Yx_fit, dtype=np.float32)
        Yy = np.asarray(Yy_fit, dtype=np.float32)

        Xs = self.scaler_X.fit_transform(X_total)
        self.reg_x.fit(Xs, Yx)
        self.reg_y.fit(Xs, Yy)
        self.fitted = True

    def predict(self, pupil_coords):
        with self.lock:
            if not self.fitted:
                return np.array([0.5, 0.5], dtype=np.float32) # [MODIFIED] 중앙값으로 안전하게 fallback

            x = np.asarray(pupil_coords, dtype=np.float32).flatten().reshape(1, -1)
            xs = self.scaler_X.transform(x)

            pred_x = float(np.asarray(self.reg_x.predict(xs)).item())
            pred_y = float(np.asarray(self.reg_y.predict(xs)).item())
            return np.array([pred_x, pred_y], dtype=np.float32)

    def movePoint(self):
        with self.lock:
            if self._tmp_X:
                self.X.extend(self._tmp_X)
                self.Y_x.extend(self._tmp_Y_x)
                self.Y_y.extend(self._tmp_Y_y)

            self._tmp_X, self._tmp_Y_x, self._tmp_Y_y = [], [], []
            self._fit_locked(tmp_only=False)
            self.matrix.movePoint()

    def isReadyToMove(self):
        return len(self._tmp_X) >= self.MIN_SAMPLES_PER_POINT

    def getCurrentPoint(self, width, height):
        return self.matrix.getCurrentPoint(width, height)

