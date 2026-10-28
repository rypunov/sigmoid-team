import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import MODEL_DIR


class EOSNormalizer:
    """
    Нормализует данные уравнения состояния.
    Использует log10 для rho и P (из-за огромного диапазона),
    и стандартизацию (Z-score) для T и E.
    """

    def __init__(self):
        self.params = {}

    def fit(self, rho: np.ndarray, T: np.ndarray, P: np.ndarray, E: np.ndarray):
        """Считает статистики (mean, std) на тренировочных данных."""
        # Добавляем 1e-12, чтобы избежать log10(0)
        log_rho = np.log10(rho + 1e-12)
        log_P = np.log10(P + 1e-12)

        self.params = {
            "rho": {"mean": float(log_rho.mean()), "std": float(log_rho.std()), "use_log": True},
            "T": {"mean": float(T.mean()), "std": float(T.std()), "use_log": False},
            "P": {"mean": float(log_P.mean()), "std": float(log_P.std()), "use_log": True},
            "E": {"mean": float(E.mean()), "std": float(E.std()), "use_log": False},
        }
        return self

    def transform(self, rho: np.ndarray, T: np.ndarray, P: np.ndarray, E: np.ndarray):
        """Нормализует данные по сохраненным параметрам."""
        rho_norm = (np.log10(rho + 1e-12) - self.params["rho"]["mean"]) / self.params["rho"]["std"]
        T_norm = (T - self.params["T"]["mean"]) / self.params["T"]["std"]
        P_norm = (np.log10(P + 1e-12) - self.params["P"]["mean"]) / self.params["P"]["std"]
        E_norm = (E - self.params["E"]["mean"]) / self.params["E"]["std"]
        return rho_norm, T_norm, P_norm, E_norm

    def inverse_transform_P(self, P_norm: np.ndarray) -> np.ndarray:
        """Переводит нормализованное P обратно в ГПа."""
        log_P = P_norm * self.params["P"]["std"] + self.params["P"]["mean"]
        return np.power(10.0, log_P) - 1e-12

    def inverse_transform_E(self, E_norm: np.ndarray) -> np.ndarray:
        """Переводит нормализованное E обратно в кДж/г."""
        return E_norm * self.params["E"]["std"] + self.params["E"]["mean"]

    def save(self, path: Path = MODEL_DIR / "normalization.json"):
        """Сохраняет параметры в JSON, чтобы другие участники могли их загрузить."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4, ensure_ascii=False)
        print(f"✅ Параметры нормализации сохранены в {path}")

    @classmethod
    def load(cls, path: Path = MODEL_DIR / "normalization.json"):
        """Загружает параметры из JSON."""
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        normalizer = cls()
        normalizer.params = params
        return normalizer


def split_and_prepare(ds: dict, random_state=42):
    """
    Делит данные на train/test (80/20), нормализует их и возвращает словарь.
    """
    n_total = len(ds["rho"])
    indices = np.arange(n_total)

    # 1. Разбиваем на train (80%) и test (20%)
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=random_state)

    # 2. Инициализируем нормализатор и фитим его ТОЛЬКО на train
    normalizer = EOSNormalizer()
    normalizer.fit(
        ds["rho"][idx_train],
        ds["T"][idx_train],
        ds["P"][idx_train],
        ds["E"][idx_train]
    )

    # 3. Нормализуем весь датасет
    rho_n, T_n, P_n, E_n = normalizer.transform(ds["rho"], ds["T"], ds["P"], ds["E"])

    # 4. Сохраняем нормализацию
    normalizer.save()

    # 5. Формируем итоговые тензоры (без val)
    data = {
        "train": {
            "rho": rho_n[idx_train], "T": T_n[idx_train],
            "P": P_n[idx_train], "E": E_n[idx_train]
        },
        "test": {
            "rho": rho_n[idx_test], "T": T_n[idx_test],
            "P": P_n[idx_test], "E": E_n[idx_test]
        },
        "normalizer": normalizer
    }
    return data


if __name__ == "__main__":
    from data_loader import load_flat_dataset

    print("Загрузка данных...")
    ds = load_flat_dataset()

    print("Разбиение и нормализация...")
    prepared = split_and_prepare(ds)

    # Печатаем только train и test (val мы убрали)
    for split in ["train", "test"]:
        print(f"{split}: {len(prepared[split]['rho'])} точек")

    print("Тест обратного преобразования:")
    P_real = prepared["normalizer"].inverse_transform_P(prepared["test"]["P"][:5])
    E_real = prepared["normalizer"].inverse_transform_E(prepared["test"]["E"][:5])
    print("P (ГПа):", P_real)
    print("E (кДж/г):", E_real)