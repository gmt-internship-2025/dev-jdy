# calibraion.py (Jetson Orin)
from __future__ import division
import cv2
import numpy as np

# 기존 의존성 유지
# from .pupil import Pupil
# from .pupil5 import Pupil
# $ python3 calibration_ui.py
# from pupil5 import Pupil
from calibration.pupil5 import Pupil


class Calibration(object):
    """
    Calibrates the pupil detection by finding the best binarization threshold.
    Jetson Orin 최적화:
      - CUDA(OpenCV) 사용 가능 시 GPU 가속
      - 히스토그램 CDF로 후보 임계치 축소
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

        # GPU 사용 가능 여부 자동 감지
        self._gpu_enabled = False
        try:
            self._gpu_enabled = (
                hasattr(cv2, "cuda")
                and cv2.cuda.getCudaEnabledDeviceCount() > 0
            )
        except Exception:
            self._gpu_enabled = False

        # GPU 전처리 필터(필요 시 초기화)
        self._gpu_gauss = None

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return (
            len(self.thresholds_left) >= self.nb_frames
            and len(self.thresholds_right) >= self.nb_frames
        )

    def threshold(self, side):
        """Returns the threshold value for the given eye.

        Argument:
            side: 0 (left) or 1 (right)
        """
        if side == 0 and len(self.thresholds_left) > 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1 and len(self.thresholds_right) > 0:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))
        # 데이터가 비어있을 때의 안전 처리
        return 50

    # ---------- 공통 유틸 ----------

    @staticmethod
    def _to_gray_u8(img):
        """입력을 그레이스케일 uint8로 변환"""
        if img is None:
            raise ValueError("eye_frame is None")
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    @staticmethod
    def _crop_border5(gray):
        """상하좌우 5px 크롭 (원 코드 동작과 동일 컨셉)"""
        h, w = gray.shape[:2]
        if h <= 10 or w <= 10:
            return gray  # 너무 작으면 스킵
        return gray[5:h - 5, 5:w - 5]

    @staticmethod
    def iris_size(frame_bin):
        """Returns the percentage of space that the iris takes up on the eye surface.

        Argument:
            frame_bin (numpy.ndarray): Binarized iris frame (uint8, 0/255)
        """
        # 이미 외곽 5px 제거가 된 상태를 기대하지만, 혹시를 대비
        frame = Calibration._crop_border5(frame_bin)
        height, width = frame.shape[:2]
        nb_pixels = height * width
        # 0이 검정(iris), 255가 흰색
        nb_whites = cv2.countNonZero(frame)
        nb_blacks = nb_pixels - nb_whites
        return nb_blacks / float(nb_pixels + 1e-6)

    @staticmethod
    def _coarse_threshold_from_cdf(gray, target_black_ratio=0.48):
        """
        히스토그램 누적분포(CDF)로 동공(검정) 비율에 맞는 임계치의 '거친 추정치'를 반환.
        gray: uint8 그레이 이미지 (이미 5px 크롭 반영)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        cdf = np.cumsum(hist) / (hist.sum() + 1e-6)
        # cdf[t] ~= 흰색 비율이 아니라 '<= t' 비율이므로,
        # threshold에 의해 0으로 가는 픽셀(검정) 비율은 roughly cdf[t]
        # 우리가 원하는 건 black_ratio ~= 0.48 이므로 cdf[t]가 0.48에 가까운 t를 찾음
        t0 = int(np.argmin(np.abs(cdf - target_black_ratio)))
        t0 = int(np.clip(t0, 5, 250))
        return t0

    # ---------- GPU 경로 ----------

    def _ensure_gpu_filters(self, ksize=(5, 5), sigma=0):
        if self._gpu_gauss is None:
            self._gpu_gauss = cv2.cuda.createGaussianFilter(
                srcType=cv2.CV_8UC1, dstType=cv2.CV_8UC1, ksize=ksize, sigma1=sigma
            )

    def _find_best_threshold_gpu(self, eye_frame):
        """
        GPU 사용: 그레이 → 5px 크롭 → 가우시안 → 후보 임계치 범위만 CUDA threshold.
        """
        gray = self._to_gray_u8(eye_frame)
        gray_c = self._crop_border5(gray)

        # 업로드 1회
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(gray_c)

        # 전처리 1회
        self._ensure_gpu_filters()
        gpu_blur = self._gpu_gauss.apply(gpu_src)

        # 먼저 CDF로 거친 임계치 추정 (CPU에서 빠르게)
        t0 = self._coarse_threshold_from_cdf(gray_c, target_black_ratio=0.48)

        # 세밀 탐색 범위 설정
        search = list(range(max(5, t0 - 15), min(250, t0 + 15) + 1, 2))
        trials = {}

        # 임계치별 바이너리 생성은 GPU, count는 CPU (작은 프레임만 다운로드)
        for thr in search:
            # THRESH_BINARY_INV: 어두운 영역(동공)을 255로 만들지/0으로 만들지?
            # 원래 iris_size는 "검정 픽셀 비율"을 기준으로 하므로, 여기서는
            # 일반 바이너리로 만들고 나중에 검정/흰정 계산에 맞춰 처리.
            # 여기서는 BINARY 사용 후 iris_size에서 '검정(0)' 비율로 계산
            _, gpu_bin = cv2.cuda.threshold(gpu_blur, thr, 255, cv2.THRESH_BINARY)
            bin_cpu = gpu_bin.download()
            trials[thr] = self.iris_size(bin_cpu)

        # 목표 동공 비율(0.48)에 가장 가까운 임계치 선택
        best_threshold, _ = min(
            trials.items(), key=(lambda p: abs(p[1] - 0.48))
        )
        return int(best_threshold)

    # ---------- CPU 경로 ----------

    def _find_best_threshold_cpu(self, eye_frame):
        """
        CPU 사용: 그레이 → 5px 크롭 → 가우시안 → CDF로 t0 → 주변 좁은 범위만 탐색
        (반복 횟수를 크게 줄여 원 코드 대비 가속)
        """
        gray = self._to_gray_u8(eye_frame)
        gray_c = self._crop_border5(gray)

        blur = cv2.GaussianBlur(gray_c, (5, 5), 0)

        # CDF 기반 거친 추정
        t0 = self._coarse_threshold_from_cdf(gray_c, target_black_ratio=0.48)

        # 세밀 탐색 범위
        search = list(range(max(5, t0 - 15), min(250, t0 + 15) + 1, 2))
        trials = {}

        for thr in search:
            _, bin_img = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
            trials[thr] = self.iris_size(bin_img)

        best_threshold, _ = min(
            trials.items(), key=(lambda p: abs(p[1] - 0.48))
        )
        return int(best_threshold)

    # ---------- 공개 API ----------

    @staticmethod
    def find_best_threshold_legacy(eye_frame):
        """
        (레거시) 원래 방식: 5~95 step 5 전체 탐색 + Pupil.image_processing 사용.
        호환성이나 비교를 위해 남겨둠.
        """
        average_iris_size = 0.48
        trials = {}
        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)
        best_threshold, _ = min(
            trials.items(), key=(lambda p: abs(p[1] - average_iris_size))
        )
        return int(best_threshold)

    def find_best_threshold(self, eye_frame):
        """
        Jetson 최적화 버전(기본): CDF로 후보 축소 + (GPU 가능 시) CUDA 가속.
        """
        try:
            if self._gpu_enabled:
                return self._find_best_threshold_gpu(eye_frame)
            else:
                return self._find_best_threshold_cpu(eye_frame)
        except Exception:
            # 예외 시 레거시 경로로 안전 폴백
            return self.find_best_threshold_legacy(eye_frame)

    def evaluate(self, eye_frame, side):
        """Improves calibration by taking into consideration the given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side: 0 (left) or 1 (right)
        """
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)

