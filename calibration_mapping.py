# calibration_mapping.py
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

class GazeMapper:
    def __init__(self):
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.is_trained = False  # 변수로 상태 관리

    def train(self, pupil_coords, screen_coords):
        """
        pupil_coords: list of (x, y) tuples from pupil detection
        screen_coords: list of (x, y) tuples corresponding to screen positions
        """
        X = np.array(pupil_coords)
        y_x = np.array([pt[0] for pt in screen_coords])
        y_y = np.array([pt[1] for pt in screen_coords])

        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        self.is_trained = True
        print("[GazeMapper] Training completed")
        
        # 모델 저장
        self.save("gaze_model.pkl")

    def predict(self, pupil_coord):
        """
        pupil_coord: (x, y) tuple from pupil detection
        return: (predicted_screen_x, predicted_screen_y)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        x = self.model_x.predict([pupil_coord])[0]
        y = self.model_y.predict([pupil_coord])[0]
        return int(x), int(y)
        
    # 모델 저장 함수
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.model_x, self.model_y), f)
            
    # 모델 불러오기 함수
    def load(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                self.model_x, self.model_y = pickle.load(f)
            self.is_trained = True
            print("[GazeMapper] 모델 불러오기 완료")
        except Exception as e:
            self.is_trained = False
            print(f"[GazeMapper] 모델 불러오기 실패: {e}")

