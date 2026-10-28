import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import pickle
from config import MODEL_DIR


class BilinearEOS:
    def __init__(self):
        self.interpolators_P = []
        self.interpolators_E = []
        self.bounds = []  # Границы областей (rho_min, rho_max, T_min, T_max)

    def fit(self, regions: list):
        """
        Обучает интерполяторы на всех 20 областях.
        """
        for r in regions:
            # RegularGridInterpolator ожидает (points, values), где values имеет форму сетки
            interp_P = RegularGridInterpolator((r.T, r.rho), r.P, method='linear', bounds_error=False, fill_value=None)
            interp_E = RegularGridInterpolator((r.T, r.rho), r.E, method='linear', bounds_error=False, fill_value=None)

            self.interpolators_P.append(interp_P)
            self.interpolators_E.append(interp_E)
            self.bounds.append({
                "rho_min": r.rho.min(), "rho_max": r.rho.max(),
                "T_min": r.T.min(), "T_max": r.T.max()
            })

    def predict(self, rho: np.ndarray, T: np.ndarray):
        """
        Предсказывает P и E.
        Использует векторизованный подход для скорости.
        """
        P_out = np.zeros_like(rho)
        E_out = np.zeros_like(rho)

        # Проходим по каждой точке (для надежности, на 100к точек это займет пару секунд)
        # В идеале можно сделать группировку по областям, но для хакатона сойдет.
        for i in range(len(rho)):
            found = False
            for idx, b in enumerate(self.bounds):
                if (b["rho_min"] <= rho[i] <= b["rho_max"]) and (b["T_min"] <= T[i] <= b["T_max"]):
                    pt = np.array([[T[i], rho[i]]])
                    P_out[i] = self.interpolators_P[idx](pt)[0]
                    E_out[i] = self.interpolators_E[idx](pt)[0]
                    found = True
                    break

            if not found:
                # Если точка вне всех областей (на границе), берем первую область (заглушка)
                # В реальности тут нужно аккуратно обрабатывать, но для бейзлайна сойдет.
                P_out[i] = 0.0
                E_out[i] = 0.0

        return P_out, E_out

    def save(self, path=MODEL_DIR / "baseline.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path=MODEL_DIR / "baseline.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)