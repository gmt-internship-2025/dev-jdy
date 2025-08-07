# my_calibrator.py

import numpy as np
import threading
import sklearn.linear_model as scireg
from sklearn.ensemble import RandomForestRegressor

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


class CalibrationMatrix:
    def __init__(self):
        self.iterator = 0
        self.points = np.array([
            [1.0, 0.5], [0.75, 0.5], [0.5, 0.5], [0.25, 0.5], [0.0, 0.5],
            [1.0, 1.0], [0.75, 1.0], [0.5, 1.0], [0.25, 1.0], [0.0, 1.0],
            [1.0, 0.0], [0.75, 0.0], [0.5, 0.0], [0.25, 0.0], [0.0, 0.0],
            [1.0, 0.75], [0.75, 0.75], [0.5, 0.75], [0.25, 0.75], [0.0, 0.75],
            [1.0, 0.25], [0.75, 0.25], [0.5, 0.25], [0.25, 0.25], [0.0, 0.25]
        ])

    def getCurrentPoint(self, width=1.0, height=1.0):
        p = self.points[self.iterator]
        return np.array([p[0] * width, p[1] * height])

    def movePoint(self):
        self.iterator = (self.iterator + 1) % len(self.points)

    def updMatrix(self, new_points):
        self.points = new_points
        self.iterator = 0


class Calibrator:
    PRECISION_LIMIT = 50
    PRECISION_STEP = 10
    ACCEPTANCE_RADIUS = 500

    def __init__(self, CALIBRATION_RADIUS=1000):
        self.X = []
        self.Y_x = []
        self.Y_y = []
        self.__tmp_X = []
        self.__tmp_Y_x = []
        self.__tmp_Y_y = []

        self.reg_x = scireg.Ridge(alpha=0.5)
        self.reg_y = scireg.Ridge(alpha=0.5)
        self.current_algorithm = "Ridge"
        self.fitted = False
        self.cv_not_set = True

        self.matrix = CalibrationMatrix()

        self.precision_limit = self.PRECISION_LIMIT
        self.precision_step = self.PRECISION_STEP
        self.acceptance_radius = int(CALIBRATION_RADIUS / 2)
        self.calibration_radius = int(CALIBRATION_RADIUS)

        self.lock = threading.Lock()
        self.fit_threads = []

    def add(self, pupil_coords, screen_coords):
        with self.lock:
            self.__tmp_X.append(np.array(pupil_coords).flatten())
            self.__tmp_Y_x.append(screen_coords[0])
            self.__tmp_Y_y.append(screen_coords[1])
            self.__launch_fit()

    def __launch_fit(self):
        thread = threading.Thread(target=self.__async_fit)
        self.fit_threads.append(thread)
        thread.start()
        self.__join_finished()

    def __join_finished(self):
        for thread in self.fit_threads:
            if not thread.is_alive():
                thread.join()

    def __async_fit(self):
        try:
            with self.lock:
                X_total = np.array(self.__tmp_X + self.X)
                Y_x_total = np.array(self.__tmp_Y_x + self.Y_x)
                Y_y_total = np.array(self.__tmp_Y_y + self.Y_y)
                self.reg_x.fit(X_total, Y_x_total)
                self.reg_y.fit(X_total, Y_y_total)
                self.fitted = True
        except Exception as e:
            print(f"[Calibrator] Fit error: {e}")

    def predict(self, pupil_coords):
        with self.lock:
            if self.fitted:
                x = np.array(pupil_coords).flatten().reshape(1, -1)
                pred_x = self.reg_x.predict(x)[0]
                pred_y = self.reg_y.predict(x)[0]
                return np.array([pred_x, pred_y])
            else:
                return np.array([0.0, 0.0])

    def movePoint(self):
        with self.lock:
            self.X += self.__tmp_X
            self.Y_x += self.__tmp_Y_x
            self.Y_y += self.__tmp_Y_y
            self.__tmp_X = []
            self.__tmp_Y_x = []
            self.__tmp_Y_y = []
            self.matrix.movePoint()

    def isReadyToMove(self):
        return len(self.__tmp_X) > 30  # 한 점에서 최소 30개 샘플

    def getCurrentPoint(self, width, height):
        return self.matrix.getCurrentPoint(width, height)

    def insideClbRadius(self, point, width, height):
        return euclidean_distance(point, self.getCurrentPoint(width, height)) < self.calibration_radius

    def insideAcptcRadius(self, point, width, height):
        return euclidean_distance(point, self.getCurrentPoint(width, height)) < self.acceptance_radius

    def unfit(self):
        self.fitted = False
        self.current_algorithm = "Ridge"

    def increase_precision(self):
        if self.acceptance_radius > self.precision_limit:
            self.acceptance_radius -= self.precision_step
        if self.calibration_radius > self.precision_limit and self.acceptance_radius < self.calibration_radius:
            self.calibration_radius -= self.precision_step

    def whichAlgorithm(self):
        return self.current_algorithm

