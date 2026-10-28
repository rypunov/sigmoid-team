import sys
from pathlib import Path
# Добавляем корень проекта в sys.path, чтобы работали импорты
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from config import REPORT_DIR  # Теперь без точки!
from data_loader import load_raw_regions, load_flat_dataset
from preprocessor import split_and_prepare
from baseline import BilinearEOS
from metrics import evaluate_model, benchmark_throughput


def run_baseline_evaluation():
    print("1. Загрузка данных...")
    ds = load_flat_dataset()
    regions = load_raw_regions()

    print("2. Подготовка данных (нормализация, сплит)...")
    prepared = split_and_prepare(ds)
    normalizer = prepared["normalizer"]
    test_data = prepared["test"]

    print("3. Обучение бейзлайна (билинейная интерполяция)...")
    baseline = BilinearEOS()
    baseline.fit(regions)

    # Обертка: переводит нормализованные входы в физические, делает predict, переводит обратно
    def baseline_predict_wrapper(rho_n, T_n):
        # 1. Нормализованные -> физические
        rho_real = np.power(10.0, rho_n * normalizer.params["rho"]["std"] + normalizer.params["rho"]["mean"]) - 1e-12
        T_real = T_n * normalizer.params["T"]["std"] + normalizer.params["T"]["mean"]

        # 2. Физические -> предсказание (в ГПа и кДж/г)
        P_real, E_real = baseline.predict(rho_real, T_real)

        # 3. Обратно в нормализованные для evaluate_model
        P_n = (np.log10(P_real + 1e-12) - normalizer.params["P"]["mean"]) / normalizer.params["P"]["std"]
        E_n = (E_real - normalizer.params["E"]["mean"]) / normalizer.params["E"]["std"]
        return P_n, E_n

    print("4. Оценка на тестовой выборке (113к точек)...")
    metrics = evaluate_model(
        baseline_predict_wrapper,
        test_data,
        normalizer,
        name="Bilinear Baseline"
    )

    print("5. Замер скорости (100k запросов)...")
    rho_rand = np.random.uniform(0.001, 10000, 100_000)
    T_rand = np.random.uniform(0.1, 1e7, 100_000)
    t_total = benchmark_throughput(baseline.predict, n_queries=100_000)
    metrics["time_100k_s"] = t_total

    # 6. Сохраняем результат
    report_path = REPORT_DIR / "baseline_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✅ Метрики бейзлайна сохранены в {report_path}")


if __name__ == "__main__":
    run_baseline_evaluation()