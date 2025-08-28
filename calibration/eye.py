# eye.py (Jetson Orin)

import math
import numpy as np
import cv2
from calibration.pupil5 import Pupil

class Eye(object):
    """
    Isolates the eye region and initializes pupil detection.
    Jetson 최적화:
      - [MODIFIED] GPU 메모리 관리 개선 및 코드 간소화
      - ROI 내 마스킹으로 계산량 최소화
    """
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    _GPU_ENABLED = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = (0, 0)
        self.center = (0, 0)
        self.pupil = None
        self.landmark_points = None
        self.blinking = 0.0

        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.landmark_points = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in points], dtype=np.int32)
        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)

    @staticmethod
    def _middle_point(p1, p2):
        return (p1.x + p2.x) // 2, (p1.y + p2.y) // 2

    def _isolate(self, frame):
        H, W = frame.shape[:2]
        margin = 5
        min_x = max(np.min(self.landmark_points[:, 0]) - margin, 0)
        max_x = min(np.max(self.landmark_points[:, 0]) + margin, W - 1)
        min_y = max(np.min(self.landmark_points[:, 1]) - margin, 0)
        max_y = min(np.max(self.landmark_points[:, 1]) + margin, H - 1)

        if max_x <= min_x or max_y <= min_y:
            self.frame = np.zeros((1, 1), dtype=np.uint8)
            return

        self.origin = (min_x, min_y)
        roi = frame[min_y:max_y + 1, min_x:max_x + 1]
        h_roi, w_roi = roi.shape[:2]

        mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        region_roi = self.landmark_points - self.origin
        cv2.fillPoly(mask, [region_roi], 255)

        # [MODIFIED] GPU/CPU bitwise_and 로직 개선
        try:
            if self._GPU_ENABLED:
                gpu_roi = cv2.cuda_GpuMat()
                gpu_roi.upload(roi)
                gpu_mask = cv2.cuda_GpuMat()
                gpu_mask.upload(mask)
                
                # frame이 BGR일 경우를 대비 (주로 gray이지만)
                if frame.ndim == 3 and frame.shape[2] == 3:
                     gpu_mask = cv2.cuda.cvtColor(gpu_mask, cv2.COLOR_GRAY2BGR)

                gpu_eye = cv2.cuda.bitwise_and(gpu_roi, gpu_mask)
                self.frame = gpu_eye.download()
            else:
                raise RuntimeError("fallback to CPU")
        except Exception:
            # CPU 폴백
            if frame.ndim == 3 and frame.shape[2] == 3:
                 mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.frame = cv2.bitwise_and(roi, mask)

        h, w = self.frame.shape[:2]
        self.center = (w / 2, h / 2)


    def _blinking_ratio(self, landmarks, points):
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot(left[0] - right[0], left[1] - right[1])
        eye_height = math.hypot(top[0] - bottom[0], top[1] - bottom[1])

        try:
            return eye_width / eye_height
        except ZeroDivisionError:
            return 0.0

