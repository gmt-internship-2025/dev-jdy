import numpy as np
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge  # 일반 선형대신 Ridge 사용해 과적합 방지

class GazeMapper:
    def __init__(self):
        # PolynomialFeatures + Ridge 회귀로 구성
        degree = 2  # 2차 다항
        self.model_x = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
        self.model_y = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
        self.is_trained = False

    def train(self, pupil_coords, screen_coords):
        X = np.array(pupil_coords)
        y_x = np.array([pt[0] for pt in screen_coords])
        y_y = np.array([pt[1] for pt in screen_coords])

        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        self.is_trained = True
        print("[GazeMapper] Training completed (Polynomial Regression)")

        self.save("gaze_model.pkl")

    def predict(self, pupil_coord):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        x = self.model_x.predict([pupil_coord])[0]
        y = self.model_y.predict([pupil_coord])[0]
        return int(x), int(y)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.model_x, self.model_y), f)

    def load(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                self.model_x, self.model_y = pickle.load(f)
            self.is_trained = True
            print("[GazeMapper] 모델 불러오기 완료 (Polynomial Regression)")
        except Exception as e:
            self.is_trained = False
            print(f"[GazeMapper] 모델 불러오기 실패: {e}")

