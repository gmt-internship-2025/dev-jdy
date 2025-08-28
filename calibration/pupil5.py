# pupil5.py (Jetson Orin)

import numpy as np
import cv2

class Pupil(object):
    """
    Detects the iris of an eye and estimates pupil position.
    Jetson 최적화:
      - [MODIFIED] CUDA 파이프라인 안정성 강화 및 리소스 관리 개선
      - CLAHE 파라미터 최적화
      - 해상도 대응 컨투어 면적 필터
    """
    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = int(threshold)
        self.x = None
        self.y = None
        
        # [MODIFIED] 생성자에서 바로 처리하도록 구조 변경
        if eye_frame is not None and eye_frame.size > 0:
            self.iris_frame = self._image_processing(eye_frame, self.threshold)
            self._detect_iris()
        else:
            # 입력 프레임이 없을 경우 안전하게 초기화
            self.iris_frame = np.zeros((1, 1), dtype=np.uint8)
            self.x = 0
            self.y = 0

    @staticmethod
    def _to_gray_u8(img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    @staticmethod
    def _image_processing_cuda(gray, threshold):
        # GPU 객체들을 명시적으로 생성하고 해제하여 리소스 누수 방지
        g, g_blur, g_clahe, g_bin, g_erode = (None,) * 5
        gauss, clahe, morph = (None,) * 3
        try:
            g = cv2.cuda_GpuMat()
            g.upload(gray)

            gauss = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
            g_blur = gauss.apply(g)
            
            # [MODIFIED] 파라미터 조정으로 과도한 대비 증폭 방지
            clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            g_clahe = clahe.apply(g_blur)

            _, g_bin = cv2.cuda.threshold(g_clahe, threshold, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, kernel, iterations=2)
            g_erode = morph.apply(g_bin)
            
            return g_erode.download()
        finally:
            # 사용한 GPU 객체들 명시적 해제
            del g, g_blur, g_clahe, g_bin, g_erode, gauss, clahe, morph

    @staticmethod
    def _image_processing_cpu(gray, threshold):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        clahe_img = clahe.apply(blur)
        
        _, binary = cv2.threshold(clahe_img, threshold, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(binary, kernel, iterations=2)
        
        return eroded

    @staticmethod
    def _image_processing(eye_frame, threshold):
        gray = Pupil._to_gray_u8(eye_frame)
        try:
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return Pupil._image_processing_cuda(gray, threshold)
            else:
                raise RuntimeError("fallback to CPU")
        except Exception:
            return Pupil._image_processing_cpu(gray, threshold)

    def _detect_iris(self):
        h, w = self.iris_frame.shape[:2]
        min_area = max(25.0, 0.002 * h * w)
        max_area = 0.25 * h * w

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        self.x = int(moments["m10"] / moments["m00"])
                        self.y = int(moments["m01"] / moments["m00"])
                        return

        # Fallback to center if no suitable contour is found
        self.x = w // 2
        self.y = h // 2

