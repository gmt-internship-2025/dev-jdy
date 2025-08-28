# eye.py (Jetson Orin)
import math
import numpy as np
import cv2

from calibration.pupil5 import Pupil

class Eye(object):
    """
    Isolates the eye region and initializes pupil detection.
    Jetson 최적화:
      - 전체 프레임 마스크 제거, ROI 내 마스킹만 수행
      - 가능 시 CUDA bitwise_and 사용 (자동 폴백)
    """

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    # [CUDA] 한 번만 체크해서 클래스 레벨 플래그로 보관
    _GPU_ENABLED = False
    try:
        _GPU_ENABLED = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        _GPU_ENABLED = False

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.blinking = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        x = (p1.x + p2.x) // 2
        y = (p1.y + p2.y) // 2
        return (int(x), int(y))

    def _isolate(self, frame, landmarks, points):
        """
        [OPT] 전체 프레임 마스킹 대신, 바운딩 박스 ROI에서만 마스크/AND를 수행.
        """
        # 원본 크기
        H, W = frame.shape[:2]

        # 랜드마크 폴리곤 (int32)
        region = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in points], dtype=np.int32)
        self.landmark_points = region

        # [OPT] 바운딩 박스 + margin 계산 및 프레임 범위로 클램프
        margin = 5
        min_x = max(int(np.min(region[:, 0]) - margin), 0)
        max_x = min(int(np.max(region[:, 0]) + margin), W - 1)
        min_y = max(int(np.min(region[:, 1]) - margin), 0)
        max_y = min(int(np.max(region[:, 1]) + margin), H - 1)

        # [OPT] 비정상 상황(너비/높이 0) 빠른 리턴
        if max_x <= min_x or max_y <= min_y:
            self.frame = frame[0:1, 0:1]  # 최소 안전 ROI
            self.origin = (0, 0)
            self.center = (0.5, 0.5)
            return

        # ROI 슬라이스
        roi = frame[min_y:max_y+1, min_x:max_x+1]

        # ROI 기준 폴리곤 좌표로 변환
        region_roi = region.copy()
        region_roi[:, 0] -= min_x
        region_roi[:, 1] -= min_y

        # ROI 크기의 마스크 생성 (단일 채널)
        h_roi, w_roi = roi.shape[:2]
        mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        cv2.fillPoly(mask, [region_roi], 255)

        # [CUDA] 가능하면 CUDA bitwise_and 사용
        eye_roi = None
        if Eye._GPU_ENABLED:
            try:
                gpu_roi = cv2.cuda_GpuMat()
                gpu_mask = cv2.cuda_GpuMat()
                gpu_roi.upload(roi)
                gpu_mask.upload(mask)
                # mask를 3채널 BGR에 적용하려면 채널 매칭 필요
                # ROI가 BGR이면 mask를 3채널로 확장해서 AND
                if roi.ndim == 3 and roi.shape[2] == 3:
                    mask3 = cv2.merge([mask, mask, mask])
                    gpu_mask3 = cv2.cuda_GpuMat()
                    gpu_mask3.upload(mask3)
                    gpu_eye = cv2.cuda.bitwise_and(gpu_roi, gpu_mask3)
                    eye_roi = gpu_eye.download()
                    # 정리
                    del gpu_mask3
                else:
                    # 그레이스케일일 경우 바로 AND
                    gpu_eye = cv2.cuda.bitwise_and(gpu_roi, gpu_mask)
                    eye_roi = gpu_eye.download()

                # 정리
                del gpu_roi, gpu_mask, gpu_eye
            except Exception:
                # 폴백: CPU AND
                if roi.ndim == 3:
                    mask3 = cv2.merge([mask, mask, mask])
                    eye_roi = cv2.bitwise_and(roi, mask3)
                else:
                    eye_roi = cv2.bitwise_and(roi, mask)
        else:
            # CPU 경로
            if roi.ndim == 3:
                mask3 = cv2.merge([mask, mask, mask])
                eye_roi = cv2.bitwise_and(roi, mask3)
            else:
                eye_roi = cv2.bitwise_and(roi, mask)

        # 결과 설정
        self.frame = eye_roi
        self.origin = (min_x, min_y)
        h, w = eye_roi.shape[:2]
        self.center = (w * 0.5, h * 0.5)

    def _blinking_ratio(self, landmarks, points):
        left  = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top    = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width  = math.hypot(left[0] - right[0], left[1] - right[1])
        eye_height = math.hypot(top[0] - bottom[0], top[1] - bottom[1])

        if eye_height == 0:
            return None
        return eye_width / eye_height

    def _analyze(self, original_frame, landmarks, side, calibration):
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)

        # [OPT] ROI 내 마스킹으로 아이소레이션
        self._isolate(original_frame, landmarks, points)

        # 캘리브레이션 업데이트
        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)

