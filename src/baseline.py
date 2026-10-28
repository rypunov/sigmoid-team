import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import pickle
from config import MODEL_DIR


class BilinearEOS:
    def __init__(self):
        self.interpolators_P = []
        self.interpolators_E = []
        self.bounds = []

    def fit(self, regions: list):
        for r in regions:
            interp_P = RegularGridInterpolator(
                (r.T, r.rho), r.P, method="linear", bounds_error=False, fill_value=None
            )
            interp_E = RegularGridInterpolator(
                (r.T, r.rho), r.E, method="linear", bounds_error=False, fill_value=None
            )

            self.interpolators_P.append(interp_P)
            self.interpolators_E.append(interp_E)
            self.bounds.append({
                "rho_min": r.rho.min(), "rho_max": r.rho.max(),
                "T_min": r.T.min(), "T_max": r.T.max()
            })

    def predict(self, rho: np.ndarray, T: np.ndarray):
        P_out = np.zeros_like(rho)
        E_out = np.zeros_like(rho)

        centers = np.array([
            (b["rho_min"] + b["rho_max"]) / 2 for b in self.bounds
        ]), np.array([
            (b["T_min"] + b["T_max"]) / 2 for b in self.bounds
        ])
        rho_centers, T_centers = centers

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
                dists = (rho[i] - rho_centers) ** 2 + (T[i] - T_centers) ** 2
                nearest = int(np.argmin(dists))
                pt = np.array([[T[i], rho[i]]])
                P_out[i] = self.interpolators_P[nearest](pt)[0]
                E_out[i] = self.interpolators_E[nearest](pt)[0]

        return P_out, E_out

    def save(self, path=MODEL_DIR / "baseline.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path=MODEL_DIR / "baseline.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)