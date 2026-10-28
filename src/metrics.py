import numpy as np
import time


def calculate_errors(y_true: np.ndarray, y_pred: np.ndarray, is_energy: bool = False):
    """
    Считает относительную ошибку.
    Для энергии (is_energy=True) добавляет 1.0 к знаменателю, чтобы избежать деления на ноль.
    """
    if is_energy:
        # E может быть отрицательной и близкой к нулю, поэтому берем |E| + 1
        error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1.0)
    else:
        # P всегда положительно и далеко от нуля
        error = np.abs(y_pred - y_true) / np.abs(y_true)
    return error


def evaluate_model(model_predict_func, ds_test: dict, normalizer, name: str = "Model"):
    """
    Единая функция оценки.
    Возвращает словарь с метриками MAPE, RMSE и временем.
    """
    # 1. Достаем нормализованные данные
    rho_n = ds_test["rho"]
    T_n = ds_test["T"]
    P_n = ds_test["P"]
    E_n = ds_test["E"]

    # 2. Делаем предсказание
    t0 = time.perf_counter()
    # Модель предсказывает нормализованные P и E
    P_pred_n, E_pred_n = model_predict_func(rho_n, T_n)
    t_pred = time.perf_counter() - t0

    # 3. Переводим предсказания обратно в физические единицы (ГПа, кДж/г)
    P_pred = normalizer.inverse_transform_P(P_pred_n)
    E_pred = normalizer.inverse_transform_E(E_pred_n)

    # Истинные значения тоже переводим в физические
    P_true = normalizer.inverse_transform_P(P_n)
    E_true = normalizer.inverse_transform_E(E_n)

    # 4. Считаем ошибки
    P_err = calculate_errors(P_true, P_pred, is_energy=False)
    E_err = calculate_errors(E_true, E_pred, is_energy=True)

    # 5. Считаем RMSE
    P_rmse = np.sqrt(np.mean((P_true - P_pred) ** 2))
    E_rmse = np.sqrt(np.mean((E_true - E_pred) ** 2))

    # 6. Считаем скорость (запросов в секунду)
    n_samples = len(rho_n)
    qps = n_samples / t_pred if t_pred > 0 else 0

    results = {
        "model_name": name,
        "P_MAPE_%": np.mean(P_err) * 100,
        "P_median_%": np.median(P_err) * 100,
        "P_p99_%": np.percentile(P_err, 99) * 100,
        "P_RMSE": P_rmse,
        "E_MAPE_%": np.mean(E_err) * 100,
        "E_median_%": np.median(E_err) * 100,
        "E_p99_%": np.percentile(E_err, 99) * 100,
        "E_RMSE": E_rmse,
        "Inference_time_s": t_pred,
        "Throughput_qps": qps
    }

    print(f"\n=== {name} ===")
    print(
        f"P: MAPE = {results['P_MAPE_%']:.4f}%, Median = {results['P_median_%']:.4f}%, p99 = {results['P_p99_%']:.4f}%")
    print(
        f"E: MAPE = {results['E_MAPE_%']:.4f}%, Median = {results['E_median_%']:.4f}%, p99 = {results['E_p99_%']:.4f}%")
    print(f"Скорость: {qps:,.0f} запросов/сек (время: {t_pred:.2f} с)")

    return results


def benchmark_throughput(model_predict_func, n_queries=100_000, rho_range=(0.001, 10000), T_range=(0.1, 1e7)):
    """
    Замеряет скорость на случайных запросах (для ТЗ 4.2).
    """
    rho_rand = np.random.uniform(rho_range[0], rho_range[1], n_queries)
    T_rand = np.random.uniform(T_range[0], T_range[1], n_queries)

    t0 = time.perf_counter()
    _ = model_predict_func(rho_rand, T_rand)
    t_total = time.perf_counter() - t0

    print(f"Скорость на {n_queries:,} запросах: {t_total:.3f} с ({n_queries / t_total:,.0f} запросов/сек)")
    return t_total