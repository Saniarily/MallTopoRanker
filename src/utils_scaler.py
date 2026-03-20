import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

class DualScaler:
    """分别对 query(10维) 与 metric(4维) 做标准化，避免混在一起。"""
    def __init__(self):
        self.q_scaler = StandardScaler()
        self.m_scaler = StandardScaler()
        self.fitted = False

    def fit(self, Q: np.ndarray, M: np.ndarray):
        self.q_scaler.fit(Q)
        self.m_scaler.fit(M)
        self.fitted = True

    def transform_q(self, Q: np.ndarray) -> np.ndarray:
        return self.q_scaler.transform(Q)

    def transform_m(self, M: np.ndarray) -> np.ndarray:
        return self.m_scaler.transform(M)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "DualScaler":
        with open(path, "rb") as f:
            return pickle.load(f)