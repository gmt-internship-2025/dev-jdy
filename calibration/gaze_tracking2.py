# gaze_tracking2.py (Jetson Orin)

import os
import cv2
import dlib
import numpy as np
from calibration.eye import Eye
from calibration.calibration import Calibration
import conf # [MODIFIED] 설정 파일 import

class GazeTracking(object):
    """
    Tracks user's gaze.
    Jetson 최적화:
      - [MODIFIED] 얼굴 검출 실패 시 마지막 위치 주변 ROI 탐색으로 속도 향상
      - dlib CNN face detector + CUDA 사용
      - 다운스케일에서 얼굴 검출 후 원본으로 스케일 복원
    """

    def __init__(self):
        self.frame = None
        self.gray_full = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        self._use_cnn = False
        self._cnn_detector = None
        self._hog_detector = dlib.get_frontal_face_detector()

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

        sp_path = os.path.join(self._model_dir, "shape_predictor_68_face_landmarks.dat")
        self._predictor = dlib.shape_predictor(sp_path)

        self._cuda_ok = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self._detect_w = 640
        self._last_face_rect = None
        self._miss_count = 0
        self._miss_tolerate = 3

    @property
    def pupils_located(self):
        try:
            return self.eye_left is not None and self.eye_right is not None and \
                   self.eye_left.pupil.x is not None and self.eye_right.pupil.x is not None
        except Exception:
            return False

    def _resize_gray_for_detection(self, gray):
        h, w = gray.shape[:2]
        if w <= self._detect_w:
            return gray, 1.0

        scale = self._detect_w / float(w)
        new_w = self._detect_w
        new_h = max(1, int(round(h * scale)))

        if self._cuda_ok:
            try:
                g = cv2.cuda_GpuMat()
                g.upload(gray)
                g_small = cv2.cuda.resize(g, (new_w, new_h), interpolation=cv2.INTER_AREA)
                gray_small = g_small.download()
                return gray_small, scale
            except Exception:
                pass

        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    @staticmethod
    def _scale_rect_to_full(r_small, inv_scale, full_w, full_h):
        x1 = max(0, int(round(r_small.left()   * inv_scale)))
        y1 = max(0, int(round(r_small.top()    * inv_scale)))
        x2 = min(full_w - 1, int(round(r_small.right() * inv_scale)))
        y2 = min(full_h - 1, int(round(r_small.bottom()* inv_scale)))
        return dlib.rectangle(x1, y1, x2, y2)

    @staticmethod
    def _pick_main_face(rects):
        if not rects: return None
        return max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

    def _analyze(self):
        # 1) 그레이스케일 변환
        if self._cuda_ok:
            try:
                g = cv2.cuda_GpuMat()
                g.upload(self.frame)
                g_gray = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY)
                self.gray_full = g_gray.download()
            except Exception:
                self.gray_full = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_full = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        H, W = self.gray_full.shape[:2]

        # 2) 얼굴 검출
        face_rect = None
        gray_small, scale = self._resize_gray_for_detection(self.gray_full)
        inv_scale = 1.0 / scale

        # [MODIFIED] 얼굴 탐색 ROI 설정
        search_roi = None
        if self._last_face_rect is not None and self._miss_count > 0:
            # 얼굴을 놓쳤다면 마지막 위치 주변으로 탐색 ROI 설정
            r = self._last_face_rect
            scale_factor = conf.FACE_ROI_SCALE
            cx, cy = r.center().x, r.center().y
            rw, rh = int(r.width() * scale_factor), int(r.height() * scale_factor)
            
            x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
            x2, y2 = min(W, cx + rw // 2), min(H, cy + rh // 2)
            
            # 탐색 영역이 너무 작거나 크면 전체 탐색으로 전환
            if (x2-x1) > 100 and (y2-y1) > 100:
                 search_roi = self.gray_full[y1:y2, x1:x2]
                 # ROI 내에서 검출 후 전체 좌표로 변환 필요
                 gray_small, scale = self._resize_gray_for_detection(search_roi)
                 inv_scale = 1.0 / scale

        try:
            detector_input = gray_small
            if self._use_cnn:
                dets = self._cnn_detector(detector_input, 0)
                rects_small = [d.rect for d in dets]
            else:
                rects_small = self._hog_detector(detector_input, 0)
            
            if search_roi is not None:
                 # ROI 기준 좌표를 전체 프레임 기준으로 변환
                 faces_full = [self._scale_rect_to_full(r, inv_scale, x2-x1, y2-y1) for r in rects_small]
                 faces_full = [dlib.rectangle(r.left()+x1, r.top()+y1, r.right()+x1, r.bottom()+y1) for r in faces_full]
            else:
                 faces_full = [self._scale_rect_to_full(r, inv_scale, W, H) for r in rects_small]
                 
        except Exception:
            faces_full = []

        # 3) 얼굴 선택 및 캐시
        if faces_full:
            face_rect = self._pick_main_face(faces_full)
            self._last_face_rect = face_rect
            self._miss_count = 0
        else:
            if self._last_face_rect is not None and self._miss_count < self._miss_tolerate:
                self._miss_count += 1
                face_rect = self._last_face_rect # 이전 위치를 임시로 사용
            else:
                self._last_face_rect = None
                self._miss_count = 0

        # 4) 랜드마크 검출 및 눈 객체 생성
        if face_rect is not None:
            try:
                landmarks = self._predictor(self.gray_full, face_rect)
                self.eye_left = Eye(self.gray_full, landmarks, 0, self.calibration)
                self.eye_right = Eye(self.gray_full, landmarks, 1, self.calibration)
                return
            except Exception:
                pass

        self.eye_left = None
        self.eye_right = None

    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)
        return None

    def pupil_right_coords(self):
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)
        return None

    def is_blinking(self):
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2.0
            return blinking_ratio > 3.8
        return False

    def annotated_frame(self):
        frame = self.frame.copy()
        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left   = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
        return frame

