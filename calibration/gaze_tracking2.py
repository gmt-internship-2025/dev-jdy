# gaze_tracking2.py (Jetson Orin)
from __future__ import division
import os
import cv2
import dlib
import numpy as np

# from eye import Eye
from calibration.eye import Eye
# from calibration import Calibration
from calibration.calibration import Calibration


class GazeTracking(object):
    """
    Tracks user's gaze.
    Jetson 최적화:
      - (가능시) dlib CNN face detector + CUDA 사용
      - 다운스케일(기본 640px)에서 얼굴 검출 후 원본으로 스케일 복원
      - GPU로 resize/cvtColor 가속 (OpenCV CUDA 빌드 시)
      - 마지막 얼굴 위치 캐싱
    """

    def __init__(self):
        self.frame = None
        self.gray_full = None  # [OPT] 원본 gray 캐시
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # --- Face detector 선택 ---
        self._use_cnn = False
        self._cnn_detector = None
        self._hog_detector = dlib.get_frontal_face_detector()

        # [CUDA] dlib이 CUDA 빌드이고 CNN 모델이 존재하면 CNN detector 사용
        try:
            cwd = os.path.abspath(os.path.dirname(__file__))
            self._model_dir = os.path.abspath(os.path.join(cwd, "trained_models"))
            cnn_model = os.path.join(self._model_dir, "mmod_human_face_detector.dat")
            if getattr(dlib, "DLIB_USE_CUDA", False) and os.path.exists(cnn_model):
                self._cnn_detector = dlib.cnn_face_detection_model_v1(cnn_model)
                self._use_cnn = True
        except Exception:
            self._use_cnn = False
            self._cnn_detector = None

        # --- Landmark predictor ---
        sp_path = os.path.join(self._model_dir, "shape_predictor_68_face_landmarks.dat")
        self._predictor = dlib.shape_predictor(sp_path)

        # --- GPU 전처리 사용 가능 여부 ---
        self._cuda_ok = False
        try:
            self._cuda_ok = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            self._cuda_ok = False

        # --- Detector 해상도 설정 ---
        self._detect_w = 640  # [OPT] 다운스케일 기준 너비
        self._last_face_rect = None  # [OPT] 최근 얼굴 위치 캐시 (dlib.rectangle)
        self._miss_count = 0
        self._miss_tolerate = 3

    @property
    def pupils_located(self):
        try:
            int(self.eye_left.pupil.x); int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x); int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    # --- 내부 유틸 ---

    def _resize_gray_for_detection(self, gray):
        """[CUDA] gray를 _detect_w 너비로 리사이즈하여 검출용 이미지와 scale 반환"""
        h, w = gray.shape[:2]
        if w <= self._detect_w:
            return gray, 1.0  # 리사이즈 불필요

        scale = self._detect_w / float(w)
        new_w = self._detect_w
        new_h = max(1, int(round(h * scale)))

        if self._cuda_ok:
            try:
                g = cv2.cuda_GpuMat()
                g.upload(gray)
                g_small = cv2.cuda.resize(g, (new_w, new_h), interpolation=cv2.INTER_AREA)
                gray_small = g_small.download()
                del g, g_small
                return gray_small, scale
            except Exception:
                pass  # 폴백

        gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray_small, scale

    @staticmethod
    def _scale_rect_to_full(r_small, inv_scale, full_w, full_h):
        """다운스케일 좌표를 원본 좌표로 복원"""
        x1 = max(0, int(round(r_small.left()   * inv_scale)))
        y1 = max(0, int(round(r_small.top()    * inv_scale)))
        x2 = min(full_w - 1, int(round(r_small.right() * inv_scale)))
        y2 = min(full_h - 1, int(round(r_small.bottom()* inv_scale)))
        return dlib.rectangle(x1, y1, x2, y2)

    def _pick_main_face(self, rects):
        """가장 큰 얼굴 선택"""
        if not rects:
            return None
        areas = [(r, (r.right() - r.left()) * (r.bottom() - r.top())) for r in rects]
        areas.sort(key=lambda x: x[1], reverse=True)
        return areas[0][0]

    # --- 핵심 분석 ---

    def _analyze(self):
        """Detect face(s) then initialize Eye objects"""
        # 1) 원본 그레이 1회 생성
        #    (가능시 CUDA로 BGR->GRAY, 아니면 CPU)
        if self._cuda_ok:
            try:
                g = cv2.cuda_GpuMat()
                g.upload(self.frame)
                g_gray = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY)
                self.gray_full = g_gray.download()
                del g, g_gray
            except Exception:
                self.gray_full = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_full = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        H, W = self.gray_full.shape[:2]

        # 2) 다운스케일 이미지 생성 (검출용)
        gray_small, scale = self._resize_gray_for_detection(self.gray_full)
        inv_scale = 1.0 / scale

        # 3) 얼굴 검출 (CNN 또는 HOG)
        faces_full = []

        try:
            if self._use_cnn and self._cnn_detector is not None:
                # CNN detector는 mmod_rectangle 반환 → .rect로 접근
                dets = self._cnn_detector(gray_small, 0)
                rects_small = [d.rect for d in dets]
            else:
                # HOG detector (upsample=0) — 이미 다운스케일이므로 업샘플 불필요
                rects_small = self._hog_detector(gray_small, 0)

            # 스케일 복원
            faces_full = [self._scale_rect_to_full(r, inv_scale, W, H) for r in rects_small]
        except Exception:
            faces_full = []

        # 4) 캐시/선택 로직
        face_rect = None
        if faces_full:
            face_rect = self._pick_main_face(faces_full)
            self._last_face_rect = face_rect
            self._miss_count = 0
        else:
            # 검출 실패 시 마지막 성공 rect를 잠시 사용
            if self._last_face_rect is not None and self._miss_count < self._miss_tolerate:
                face_rect = self._last_face_rect
                self._miss_count += 1
            else:
                self._last_face_rect = None
                self._miss_count = 0

        # 5) 랜드마크 & Eye
        if face_rect is not None:
            try:
                landmarks = self._predictor(self.gray_full, face_rect)
                # Eye는 그레이 프레임을 기대(원 코드와 동일 패턴)
                self.eye_left  = Eye(self.gray_full, landmarks, 0, self.calibration)
                self.eye_right = Eye(self.gray_full, landmarks, 1, self.calibration)
                return
            except Exception:
                pass

        # 실패 시 None 세팅
        self.eye_left = None
        self.eye_right = None

    # --- 공개 API ---

    def refresh(self, frame):
        """Update frame and analyze."""
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        # [원 코드의 수정내용 유지] 눈 폭 기준 정규화
        if not self.pupils_located:
            return None

        left_w  = self.eye_left.landmark_points[:, 0].ptp()
        right_w = self.eye_right.landmark_points[:, 0].ptp()
        if left_w == 0 or right_w == 0:
            return None

        pl = (self.eye_left.pupil.x  - self.eye_left.center[0])  / (left_w  / 2.0)
        pr = (self.eye_right.pupil.x - self.eye_right.center[0]) / (right_w / 2.0)

        pl = (pl + 1.0) / 2.0
        pr = (pr + 1.0) / 2.0
        return (pl + pr) / 2.0

    def vertical_ratio(self):
        if self.pupils_located:
            denom_l = (self.eye_left.center[1]  * 2.0 - 10.0)
            denom_r = (self.eye_right.center[1] * 2.0 - 10.0)
            if denom_l == 0 or denom_r == 0:
                return None
            pl = self.eye_left.pupil.y  / denom_l
            pr = self.eye_right.pupil.y / denom_r
            return (pl + pr) / 2.0

    # 시선 방향 판정 로직 유지
    def is_right(self):
        hr = self.horizontal_ratio()
        return hr is not None and hr < 0.45

    def is_left(self):
        hr = self.horizontal_ratio()
        return hr is not None and hr > 0.55

    def is_center(self):
        hr = self.horizontal_ratio()
        return hr is not None and 0.45 <= hr <= 0.55

    def is_blinking(self):
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2.0
            return blinking_ratio > 3.8

    def annotated_frame(self):
        frame = self.frame.copy()
        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left   = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5,  y_left), (x_left + 5,  y_left), color)
            cv2.line(frame, (x_left,      y_left - 5), (x_left,      y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right,     y_right - 5), (x_right,     y_right + 5), color)
        return frame

