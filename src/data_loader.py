import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, N_FILES


class EOSRegion:
    """Класс для хранения одной области (OblXX.txt)"""

    def __init__(self, name: str, rho: np.ndarray, T: np.ndarray, P: np.ndarray, E: np.ndarray, grid_type: str):
        self.name = name
        self.rho = rho  # 1D массив уникальных rho
        self.T = T  # 1D массив уникальных T
        self.P = P  # 2D массив (n_T, n_rho)
        self.E = E  # 2D массив (n_T, n_rho)
        self.grid_type = grid_type

    def __repr__(self):
        return f"EOSRegion({self.name}, rho: {len(self.rho)}, T: {len(self.T)})"


def load_raw_regions() -> list[EOSRegion]:
    """
    Читает 20 файлов Obl01.txt..Obl20.txt из DATA_DIR.
    """
    regions = []

    for i in range(1, N_FILES + 1):
        file_name = f"Obl{i:02d}.txt"
        file_path = DATA_DIR / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Файл {file_path} не найден.")

        # 1. Читаем заголовок, чтобы понять тип сетки
        grid_type = "log-log"  # по умолчанию
        with open(file_path, 'r') as f:
            for line in f:
                if "Grid type:" in line:
                    grid_type = line.split(":")[1].strip()
                    break

        # 2. Читаем данные
        # comment='#' игнорирует строки с заголовками
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, names=['rho', 'T', 'P', 'e'])

        # 3. Восстанавливаем 2D сетку
        rho_uniq = np.sort(df['rho'].unique())
        T_uniq = np.sort(df['T'].unique())

        # Используем pivot_table, чтобы разложить 1D в 2D
        # index='T', columns='rho' -> на выходе матрица (n_T, n_rho)
        P_2d = df.pivot_table(index='T', columns='rho', values='P').values
        E_2d = df.pivot_table(index='T', columns='rho', values='e').values

        regions.append(EOSRegion(
            name=f"Obl{i:02d}",
            rho=rho_uniq,
            T=T_uniq,
            P=P_2d,
            E=E_2d,
            grid_type=grid_type
        ))

    return regions


def load_flat_dataset() -> dict:
    """
    Собирает все области в один плоский датасет (1D массивы).
    """
    regions = load_raw_regions()

    all_rho, all_T, all_P, all_E, all_region = [], [], [], [], []

    for idx, r in enumerate(regions):
        # np.meshgrid создает 2D сетки из 1D массивов
        rho_grid, T_grid = np.meshgrid(r.rho, r.T)

        all_rho.append(rho_grid.ravel())
        all_T.append(T_grid.ravel())
        all_P.append(r.P.ravel())
        all_E.append(r.E.ravel())
        all_region.append(np.full(r.P.size, idx))

    return {
        "rho": np.concatenate(all_rho),
        "T": np.concatenate(all_T),
        "P": np.concatenate(all_P),
        "E": np.concatenate(all_E),
        "region": np.concatenate(all_region),
    }

if __name__ == "__main__":
    print("Начинаем тест загрузки данных...")
    ds = load_flat_dataset()
    print(f"Всего точек: {len(ds['rho']):,}")
    print(f"Диапазон rho: {ds['rho'].min():.4g} .. {ds['rho'].max():.4g}")
    print(f"Диапазон T:   {ds['T'].min():.4g} .. {ds['T'].max():.4g}")
    print(f"Диапазон P:   {ds['P'].min():.4g} .. {ds['P'].max():.4g}")