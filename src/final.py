"""
Модуль 4: Инференс и упаковка (с использованием существующих функций из metrics.py).
"""
import sys
import os
import json
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd

# Импортируем ВАШИ готовые функции оценки
from src.metrics import evaluate_model, benchmark_throughput
from src.wrapper import EOSModelWrapper  # Импортируем нашу обертку (п. 4.4)

from src.config import SEED, DEVICE
from src.data_loader import load_flat_dataset
from src.preprocessor import split_and_prepare
from MLP import MLP


def main():
    print("--- 1. ЗАГРУЗКА ДАННЫХ ---")
    ds = load_flat_dataset()
    prepared = split_and_prepare(ds)
    normalizer = prepared["normalizer"]

    # Создаем папки
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # ПУНКТ 4.4: КЛАСС-ОБЕРТКА
    # =====================================================================
    print("\n--- 4.4. ДЕМОНСТРАЦИЯ КЛАССА-ОБЕРТКИ ---")

    # Загружаем модель (веса)
    model = MLP(in_dim=2, out_dim=2, width=128, depth=5).to(DEVICE)
    model.load_state_dict(torch.load("vanilla_model.pth", map_location=DEVICE))

    # Оборачиваем её
    wrapper = EOSModelWrapper(model, normalizer, model_type="pytorch")

    # Тестируем методы обертки на одной точке
    rho_phys = np.array([2.7])
    T_phys = np.array([300.0])

    P_pred, E_pred = wrapper.predict(rho_phys, T_phys)
    cs_pred = wrapper.sound_speed(rho_phys, T_phys)
    flags = wrapper.check_physics(rho_phys, T_phys)

    print(f"Тест обертки (rho = 2.7 г/см^3, T = 300 K):")
    print(f"  P  = {P_pred[0]:.4f} ГПа")
    print(f"  E  = {E_pred[0]:.4f} кДж/г")
    print(f"  cs = {cs_pred[0]:.4f} км/с")
    print(f"  Нарушения: {flags}")

    # =====================================================================
    # 4.1, 4.2, 4.3: ОЦЕНКА, РАЗМЕР, СКОРОСТЬ, TORCHSCRIPT
    # =====================================================================

    # Готовим функцию предсказания для ВАШЕЙ evaluate_model
    # Она должна принимать нормализованные rho и T и возвращать нормализованные P и E
    def model_predict_func(rho_n, T_n):
        x = torch.stack([
            torch.tensor(rho_n, dtype=torch.float32),
            torch.tensor(T_n, dtype=torch.float32)
        ], dim=1).to(DEVICE)
        with torch.no_grad():
            pred = model(x).cpu().numpy()
        return pred[:, 0], pred[:, 1]

    # Ваша функция оценки (4.1, 4.2)
    print("\n--- ОЦЕНКА VANILLA (используем ваш metrics.py) ---")
    results_vanilla = evaluate_model(model_predict_func, prepared["test"], normalizer, name="Vanilla MLP")

    # Тест на 100 000 запросах (4.2)
    print("\n--- ТЕСТ СКОРОСТИ (100 000 запросов) ---")
    t_total = benchmark_throughput(model_predict_func, n_queries=100_000)

    # Экспорт в TorchScript (4.3)
    print("\n--- ЭКСПОРТ В TORCHSCRIPT ---")
    model.eval()
    dummy_input = torch.randn(1, 2, device=DEVICE)
    traced_model = torch.jit.trace(model, dummy_input)
    ts_path = "vanilla_scripted.pt"
    traced_model.save(ts_path)
    size_mb = os.path.getsize(ts_path) / (1024 ** 2)
    print(f"Модель сохранена в {ts_path}, размер: {size_mb:.4f} МБ")

    # =====================================================================
    # 4.5: ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ
    # =====================================================================
    print("\n--- 4.5. ИТОГОВАЯ ТАБЛИЦА ---")

    # Данные из вашего отчета для Baseline
    baseline_data = {
        "Model": "Baseline (Интерполяция)",
        "MAPE P (%)": 1.0108,
        "MAPE E (%)": 0.1885,
        "R2 (P)": 0.9998,
        "Speed (QPS)": 9000,
        "Size (MB)": 4.36,
        "Physics Violations": "Yes"
    }

    # Данные Vanilla (берем из results_vanilla)
    vanilla_data = {
        "Model": "Vanilla MLP",
        "MAPE P (%)": results_vanilla["P_MAPE_%"],
        "MAPE E (%)": results_vanilla["E_MAPE_%"],
        "R2 (P)": results_vanilla["P_R2"],
        "Speed (QPS)": results_vanilla["Throughput_qps"],
        "Size (MB)": round(size_mb, 4),
        "Physics Violations": f"{flags['total_violations']} points"
    }

    # Собираем в DataFrame
    df = pd.DataFrame([baseline_data, vanilla_data])
    df.to_csv("comparison_table.csv", index=False)

    print(df.to_string(index=False))
    print("\nТаблица сохранена в comparison_table.csv")


if __name__ == "__main__":
    main()