## pupil5.py (Jetson Orin)

import numpy as np
import cv2


class Pupil(object):
    """
    Detects the iris of an eye and estimates pupil position.
    Jetson 최적화:
      - (가능 시) OpenCV CUDA로 전처리 가속
      - CLAHE로 대비 보정
      - 해상도 대응 컨투어 면적 필터
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = int(threshold)
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def _to_gray_u8(img):
        """[OPT] 입력을 단일채널 uint8로 일관화"""
        if img is None:
            raise ValueError("eye_frame is None")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    @staticmethod
    def image_processing(eye_frame, threshold):
        """
        [변경] 가능 시 CUDA 경로로 전처리 가속:
          - GaussianBlur -> CLAHE -> THRESH_BINARY_INV -> Erode
        실패/미지원 시 원래 CPU 파이프라인(개선판) 사용
        """
        gray = Pupil._to_gray_u8(eye_frame)

        # --- CUDA 사용 가능 여부 ---
        cuda_ok = False
        try:
            cuda_ok = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            cuda_ok = False

        if cuda_ok:
            try:
                # [CUDA] 업로드
                g = cv2.cuda_GpuMat()
                g.upload(gray)

                # [CUDA] GaussianBlur (bilateral보다 가볍고 CUDA 지원 확실)
                gauss = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
                g_blur = gauss.apply(g)

                # [CUDA] CLAHE (equalizeHist 대체)
                clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                g_eq = clahe.apply(g_blur)

                # [CUDA] THRESH_BINARY_INV (어두운 동공을 흰색으로 만들려면 INV가 직관적이나,
                # 본 코드의 iris_size/컨투어 로직은 검정(0) 기반이므로 일반 BINARY 사용 후
                # 이후 계산에서 대응하거나, 여기서는 원 코드에 맞춰 INV 유지)
                _, g_bin = cv2.cuda.threshold(g_eq, threshold, 255, cv2.THRESH_BINARY_INV)

                # [CUDA] Erode (노이즈 제거)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mf = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, kernel, iterations=2)
                g_erode = mf.apply(g_bin)

                # [CUDA->CPU] 컨투어용으로 최종 바이너리만 다운로드
                new_frame = g_erode.download()

                # 메모리 정리
                del g, g_blur, g_eq, g_bin, g_erode, gauss, clahe, mf

                return new_frame

            except Exception:
                # CUDA 실패 시 CPU 폴백
                pass

        # --- CPU 폴백 경로 (개선판) ---
        # Bilateral은 무겁고 Jetson에서 느릴 수 있어 Gaussian으로 대체 (원하면 주석 해제)
        # new_frame = cv2.bilateralFilter(gray, 10, 15, 15)
        new_frame = cv2.GaussianBlur(gray, (5, 5), 0)

        # CLAHE (equalizeHist 대체)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        new_frame = clahe.apply(new_frame)

        # 침식으로 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.erode(new_frame, kernel, iterations=2)

        # 동공을 더 잘 분리하기 위해 흑백 반전 이진화
        _, new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY_INV)

        return new_frame

    def detect_iris(self, eye_frame):
        """
        컨투어에서 가장 타당한 동공을 선택하고, 중심 모멘트로 (x, y) 추정.
        실패 시 중앙 fallback.
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        # [OPT] 해상도 기반 면적 제한
        h, w = self.iris_frame.shape[:2]
        area_img = float(h * w)
        min_area = max(25.0, 0.002 * area_img)   # 0.2% 이상
        max_area = 0.25 * area_img               # 25% 이하

        # [OPT] 외곽만 탐색
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 큰 것부터 검사
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                a = cv2.contourArea(contour)
                if a < min_area or a > max_area:
                    continue
                m = cv2.moments(contour)
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    self.x, self.y = cx, cy
                    return

        # 실패 시 중앙 fallback
        self.x = w // 2
        self.y = h // 2

