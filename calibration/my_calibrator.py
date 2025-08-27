# my_calibrator.py (Jetson Orin)

import numpy as np
import threading

# --- [GPU] cuML이 있으면 GPU Ridge 사용, 없으면 scikit-learn으로 폴백 ---
_CUML_AVAILABLE = False
try:
    from cuml.linear_model import Ridge as _cuRidge        # GPU
    from cuml.preprocessing import StandardScaler as _cuScaler
    _CUML_AVAILABLE = True
except Exception:
    from sklearn.linear_model import Ridge as _skRidge      # CPU
    from sklearn.preprocessing import StandardScaler as _skScaler
    _CUML_AVAILABLE = False


def euclidean_distance(p1, p2):
    # [OPT] float32 일관화
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    return np.linalg.norm(p1 - p2)


class CalibrationMatrix:
    def __init__(self):
        self.iterator = 0
        # 5x5 그리드 정의 (상단=0.0, 하단=1.0)
        xs = [0.00, 0.25, 0.50, 0.75, 1.00]
        ys = [0.00, 0.25, 0.50, 0.75, 1.00]
        points = []
        for row_idx, y in enumerate(ys):
            if row_idx % 2 == 0:
                scan_x = xs  # 좌 -> 우
            else:
                scan_x = xs[::1]  # 좌 -> 우
            for x in scan_x:
                points.append([x, y])
        self.points = np.array(points, dtype=np.float32)  # shape: (25,2)

    def getCurrentPoint(self, width=1.0, height=1.0):
        p = self.points[self.iterator]
        return np.array([p[0] * width, p[1] * height], dtype=np.float32)

    def movePoint(self):
        self.iterator = (self.iterator + 1) % len(self.points)

    def updMatrix(self, new_points):
        self.points = np.asarray(new_points, dtype=np.float32)
        self.iterator = 0


class Calibrator:
    # 기존 파라미터 유지
    PRECISION_LIMIT = 50
    PRECISION_STEP = 10
    ACCEPTANCE_RADIUS = 500

    # [BATCH] 배치 학습 트리거 크기 (한 포인트에서 최소 샘플 수)
    MIN_SAMPLES_PER_POINT = 30

    def __init__(self, CALIBRATION_RADIUS=1000):
        # 누적 데이터 (전체)
        self.X   = []  # list of np.array(shape=(2,))
        self.Y_x = []  # list of float
        self.Y_y = []  # list of float

        # 임시 버퍼(현재 포인트에서만)
        self.__tmp_X   = []
        self.__tmp_Y_x = []
        self.__tmp_Y_y = []

        # [GPU] 백엔드 선택 및 회귀기/스케일러 준비
        if _CUML_AVAILABLE:
            self.reg_x = _cuRidge(alpha=0.5)
            self.reg_y = _cuRidge(alpha=0.5)
            self.scaler_X = _cuScaler(with_mean=True, with_std=True)
            self._backend = "GPU(cuML)"
        else:
            self.reg_x = _skRidge(alpha=0.5)
            self.reg_y = _skRidge(alpha=0.5)
            self.scaler_X = _skScaler(with_mean=True, with_std=True)
            self._backend = "CPU(sklearn)"

        self.current_algorithm = "Ridge"
        self.fitted = False
        self.cv_not_set = True

        self.matrix = CalibrationMatrix()

        self.precision_limit = self.PRECISION_LIMIT
        self.precision_step = self.PRECISION_STEP
        self.acceptance_radius = int(CALIBRATION_RADIUS / 2)
        self.calibration_radius = int(CALIBRATION_RADIUS)

        self.lock = threading.Lock()

    # --- 데이터 수집 ---

    def add(self, pupil_coords, screen_coords):
        """
        샘플 추가(현재 포인트 버퍼에만). 즉시 학습하지 않고 모았다가 배치 학습.
        """
        pc = np.asarray(pupil_coords, dtype=np.float32).flatten()
        scx = np.float32(screen_coords[0])
        scy = np.float32(screen_coords[1])

        with self.lock:
            self.__tmp_X.append(pc)
            self.__tmp_Y_x.append(scx)
            self.__tmp_Y_y.append(scy)
            # [BATCH] 충분히 모였고, 이전에 학습된 모델이 없다면 미리 한 번 학습해둠(선택)
            if not self.fitted and len(self.__tmp_X) >= self.MIN_SAMPLES_PER_POINT:
                self.__fit_locked(tmp_only=True)

    # --- 학습(배치) ---

    def __fit_locked(self, tmp_only=False):
        """
        잠금 상태에서 호출. tmp_only=True면 현재 포인트 버퍼만으로 1차 학습.
        """
        if tmp_only:
            X_total = np.asarray(self.__tmp_X, dtype=np.float32)
            Yx      = np.asarray(self.__tmp_Y_x, dtype=np.float32)
            Yy      = np.asarray(self.__tmp_Y_y, dtype=np.float32)
        else:
            # 전체 데이터(기존 + 현재 포인트)로 재학습
            X_total = np.asarray(self.X + self.__tmp_X, dtype=np.float32) if (self.X or self.__tmp_X) else None
            Yx      = np.asarray(self.Y_x + self.__tmp_Y_x, dtype=np.float32) if (self.Y_x or self.__tmp_Y_x) else None
            Yy      = np.asarray(self.Y_y + self.__tmp_Y_y, dtype=np.float32) if (self.Y_y or self.__tmp_Y_y) else None

        if X_total is None or len(X_total) == 0:
            return

        # [OPT] 스케일러: X만 표준화
        Xs = self.scaler_X.fit_transform(X_total)

        # 회귀 학습
        self.reg_x.fit(Xs, Yx)
        self.reg_y.fit(Xs, Yy)
        self.fitted = True

    # --- 예측 ---

    def predict(self, pupil_coords):
        with self.lock:
            if not self.fitted:
                return np.array([0.0, 0.0], dtype=np.float32)
            x = np.asarray(pupil_coords, dtype=np.float32).flatten().reshape(1, -1)
            xs = self.scaler_X.transform(x)
            # cuML은 cupy/numpy를 반환할 수 있음 → np.asarray로 일관화
            pred_x = float(np.asarray(self.reg_x.predict(xs)).ravel()[0])
            pred_y = float(np.asarray(self.reg_y.predict(xs)).ravel()[0])
            return np.array([pred_x, pred_y], dtype=np.float32)

    # --- 포인트 전환(배치 확정) ---

    def movePoint(self):
        """
        현재 포인트의 버퍼를 메인 데이터에 병합하고, 그 시점에 배치 학습 수행.
        """
        with self.lock:
            # 메인 데이터에 병합 (리스트 확장)
            if self.__tmp_X:
                self.X   += self.__tmp_X
                self.Y_x += self.__tmp_Y_x
                self.Y_y += self.__tmp_Y_y

            # 임시 버퍼 초기화
            self.__tmp_X   = []
            self.__tmp_Y_x = []
            self.__tmp_Y_y = []

            # [BATCH] 전체 데이터로 재학습 (GPU면 GPU에서)
            self.__fit_locked(tmp_only=False)

            # 다음 캘리브레이션 포인트로 이동
            self.matrix.movePoint()

    # --- 보조 메서드/속성 ---

    def isReadyToMove(self):
        # [BATCH] 한 점에서 최소 N개 수집됐는지
        return len(self.__tmp_X) >= self.MIN_SAMPLES_PER_POINT

    def getCurrentPoint(self, width, height):
        return self.matrix.getCurrentPoint(width, height)

    def insideClbRadius(self, point, width, height):
        return euclidean_distance(point, self.getCurrentPoint(width, height)) < self.calibration_radius

    def insideAcptcRadius(self, point, width, height):
        return euclidean_distance(point, self.getCurrentPoint(width, height)) < self.acceptance_radius

    def unfit(self):
        self.fitted = False
        self.current_algorithm = "Ridge"

    def increase_precision(self):
        if self.acceptance_radius > self.precision_limit:
            self.acceptance_radius -= self.precision_step
        if self.calibration_radius > self.precision_limit and self.acceptance_radius < self.calibration_radius:
            self.calibration_radius -= self.precision_step

    def whichAlgorithm(self):
        # 백엔드 정보도 같이 보고 싶을 때 유용
        return f"{self.current_algorithm} @ {self._backend}"

