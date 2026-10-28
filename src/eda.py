import sys
import os
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы работали импорты
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIG_DIR  # Импорт без точки!

from data_loader import load_flat_dataset, load_raw_regions


def plot_distributions(ds: dict):
    """
    Рисует 4 гистограммы: rho, T, P, E (в логарифмическом масштабе, где нужно).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # rho
    axes[0, 0].hist(np.log10(ds['rho']), bins=100, color='blue', alpha=0.7)
    axes[0, 0].set_title('Распределение log10(rho) [g/cm3]')

    # T
    axes[0, 1].hist(np.log10(ds['T']), bins=100, color='green', alpha=0.7)
    axes[0, 1].set_title('Распределение log10(T) [K]')

    # P
    axes[1, 0].hist(np.log10(ds['P']), bins=100, color='red', alpha=0.7)
    axes[1, 0].set_title('Распределение log10(P) [GPa]')

    # E
    axes[1, 1].hist(ds['E'], bins=100, color='purple', alpha=0.7)
    axes[1, 1].set_title('Распределение E [kJ/g]')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_distributions.png", dpi=150)
    plt.show()


def plot_sections(ds: dict, regions: list):
    """
    Рисует изотермы и изохоры (сечения).
    """
    # Берем пару областей для примера (например, Obl01 и Obl06)
    # В реальности можно построить сеткой по всем T и rho

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Изотерма (фиксируем T)
    # Найдем точки, где T примерно равна 1000 K
    mask_T = np.isclose(ds['T'], 1000, rtol=0.01)
    # Сортируем по rho для красивой линии
    sort_idx = np.argsort(ds['rho'][mask_T])
    axes[0].plot(ds['rho'][mask_T][sort_idx], ds['P'][mask_T][sort_idx], 'b.-', label='T = 1000 K')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('rho [g/cm3]')
    axes[0].set_ylabel('P [GPa]')
    axes[0].set_title('Изотерма P(rho)')
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # 2. Изохора (фиксируем rho)
    # Найдем точки, где rho примерно равна 1.0
    mask_rho = np.isclose(ds['rho'], 1.0, rtol=0.01)
    sort_idx = np.argsort(ds['T'][mask_rho])
    axes[1].plot(ds['T'][mask_rho][sort_idx], ds['P'][mask_rho][sort_idx], 'r.-', label='rho = 1.0 g/cm3')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('T [K]')
    axes[1].set_ylabel('P [GPa]')
    axes[1].set_title('Изохора P(T)')
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_sections.png", dpi=150)
    plt.show()


def check_physics(ds: dict):
    """
    Проверяет физические ограничения на данных.
    Считает долю точек, где нарушается dP/drho > 0.
    """
    # Простой пример: проверяем знак P
    print(f"Доля точек с P < 0: {(ds['P'] < 0).mean() * 100:.4f}%")
    print(f"Доля точек с P > 1e-10: {(ds['P'] > 1e-10).mean() * 100:.4f}%")

    # Можно добавить проверку на c_v > 0 (требует вычисления производных)
    # Но для EDA это слишком сложно. Оставим для PINN.