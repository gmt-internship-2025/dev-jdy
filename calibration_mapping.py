# calibration_mapping.py
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor

class GazeMapper:
    def __init__(self):
        # 비선형 MLP 모델 사용
        self.model_x = MLPRegressor(hidden_layer_sizes=(64, 64),
                                    activation='relu',
                                    solver='adam',
                                    max_iter=500,
                                    random_state=42)
        self.model_y = MLPRegressor(hidden_layer_sizes=(64, 64),
                                    activation='relu',
                                    solver='adam',
                                    max_iter=500,
                                    random_state=42)
        self.is_trained = False

    def train(self, pupil_coords, screen_coords):
        X = np.array(pupil_coords)
        y_x = np.array([pt[0] for pt in screen_coords])
        y_y = np.array([pt[1] for pt in screen_coords])

        print("[GazeMapper] MLP 모델 학습 중...")
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        self.is_trained = True
        print("[GazeMapper] MLP 학습 완료")

        self.save("gaze_model.pkl")

    def predict(self, pupil_coord):
        if not self.is_trained:
            raise RuntimeError("모델이 아직 학습되지 않았습니다.")

        x = self.model_x.predict([pupil_coord])[0]
        y = self.model_y.predict([pupil_coord])[0]
        return int(x), int(y)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.model_x, self.model_y), f)
        print(f"[GazeMapper] 모델 저장 완료: {filepath}")

    def load(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                self.model_x, self.model_y = pickle.load(f)
            self.is_trained = True
            print(f"[GazeMapper] 모델 불러오기 완료: {filepath}")
        except Exception as e:
            self.is_trained = False
            print(f"[GazeMapper] 모델 불러오기 실패: {e}")

