import numpy as np
import time


def calculate_errors(y_true: np.ndarray, y_pred: np.ndarray, is_energy: bool = False):
    """Вычисляет относительную ошибку между истинными и предсказанными значениями.

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.
        is_energy (bool, optional): Если True, используется знаменатель
            `|y_true| + 1.0` (для энергии); иначе — `|y_true|`.
            По умолчанию False.

    Returns:
        np.ndarray: Массив относительных ошибок той же формы, что и входные массивы.
    """
    if is_energy:
        error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1.0)
    else:
        error = np.abs(y_pred - y_true) / np.abs(y_true)
    return error


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Вычисляет коэффициент детерминации R².

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        float: Значение R². Если общая сумма квадратов равна нулю,
            возвращается `float("nan")`.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def evaluate_model(model_predict_func, ds_test: dict, normalizer, name: str = "Model"):
    """Оценивает модель уравнения состояния на тестовых данных.

    Выполняет предсказание, переводит нормализованные значения в физические,
    вычисляет метрики ошибок для давления и энергии (MAPE, медиана, p99,
    RMSE, NRMSE, R², log R²), а также время инференса и пропускную способность.
    Выводит метрики в стандартный поток вывода.

    Args:
        model_predict_func (callable): Функция предсказания, принимающая
            нормализованные `rho` и `T` (форма (N,)) и возвращающая кортеж
            нормализованных `(P_pred, E_pred)`.
        ds_test (dict): Словарь тестовых данных с ключами 'rho', 'T', 'P', 'E'
            (нормализованные одномерные массивы).
        normalizer (EOSNormalizer): Объект нормализации с методами
            `inverse_transform_P` и `inverse_transform_E`.
        name (str, optional): Имя модели для вывода и поля `model_name`.
            По умолчанию "Model".

    Returns:
        dict: Словарь с метриками модели, включая `model_name`, ошибки и
            скорости для давления и энергии, `Inference_time_s`,
            `Throughput_qps`.
    """
    rho_n = ds_test["rho"]
    T_n = ds_test["T"]
    P_n = ds_test["P"]
    E_n = ds_test["E"]

    t0 = time.perf_counter()
    P_pred_n, E_pred_n = model_predict_func(rho_n, T_n)
    t_pred = time.perf_counter() - t0

    P_pred = normalizer.inverse_transform_P(P_pred_n)
    E_pred = normalizer.inverse_transform_E(E_pred_n)
    P_true = normalizer.inverse_transform_P(P_n)
    E_true = normalizer.inverse_transform_E(E_n)

    P_err = calculate_errors(P_true, P_pred, is_energy=False)
    E_err = calculate_errors(E_true, E_pred, is_energy=True)

    P_rmse = np.sqrt(np.mean((P_true - P_pred) ** 2))
    E_rmse = np.sqrt(np.mean((E_true - E_pred) ** 2))

    P_range = P_true.max() - P_true.min()
    E_range = E_true.max() - E_true.min()
    P_nrmse = P_rmse / P_range if P_range > 0 else float("nan")
    E_nrmse = E_rmse / E_range if E_range > 0 else float("nan")

    P_r2 = calculate_r2(P_true, P_pred)
    E_r2 = calculate_r2(E_true, E_pred)
    P_r2_log = calculate_r2(np.log1p(np.abs(P_true)), np.log1p(np.abs(P_pred)))
    E_r2_log = calculate_r2(
        np.sign(E_true) * np.log1p(np.abs(E_true)),
        np.sign(E_pred) * np.log1p(np.abs(E_pred)),
    )

    qps = len(rho_n) / t_pred if t_pred > 0 else 0

    results = {
        "model_name": name,
        "P_MAPE_%": np.mean(P_err) * 100,
        "P_median_%": np.median(P_err) * 100,
        "P_p99_%": np.percentile(P_err, 99) * 100,
        "P_RMSE": P_rmse,
        "P_NRMSE": P_nrmse,
        "P_R2": P_r2,
        "P_R2_log": P_r2_log,
        "E_MAPE_%": np.mean(E_err) * 100,
        "E_median_%": np.median(E_err) * 100,
        "E_p99_%": np.percentile(E_err, 99) * 100,
        "E_RMSE": E_rmse,
        "E_NRMSE": E_nrmse,
        "E_R2": E_r2,
        "E_R2_log": E_r2_log,
        "Inference_time_s": t_pred,
        "Throughput_qps": qps,
    }

    print(f"\n=== {name} ===")
    print(
        f"P: MAPE = {results['P_MAPE_%']:.4f}%, Median = {results['P_median_%']:.4f}%, "
        f"p99 = {results['P_p99_%']:.4f}%, RMSE = {P_rmse:.4e}, NRMSE = {P_nrmse*100:.4f}%, "
        f"R² = {P_r2:.6f} (log R² = {P_r2_log:.6f})"
    )
    print(
        f"E: MAPE = {results['E_MAPE_%']:.4f}%, Median = {results['E_median_%']:.4f}%, "
        f"p99 = {results['E_p99_%']:.4f}%, RMSE = {E_rmse:.4e}, NRMSE = {E_nrmse*100:.4f}%, "
        f"R² = {E_r2:.6f} (log R² = {E_r2_log:.6f})"
    )
    print(f"Скорость: {qps:,.0f} запросов/сек (время: {t_pred:.2f} с)")

    return results


def benchmark_throughput(model_predict_func, n_queries=100_000, rho_range=(0.001, 10000), T_range=(0.1, 1e7)):
    """Замеряет пропускную способность функции предсказания на случайных данных.

    Генерирует `n_queries` случайных точек в заданных диапазонах `rho` и `T`,
    выполняет предсказание и измеряет общее время. Выводит время и число
    запросов в секунду.

    Args:
        model_predict_func (callable): Функция предсказания, принимающая
            массивы `rho` и `T` формы (N,).
        n_queries (int, optional): Количество случайных запросов.
            По умолчанию 100_000.
        rho_range (tuple[float, float], optional): Диапазон случайных значений
            плотности `(min, max)`. По умолчанию (0.001, 10000).
        T_range (tuple[float, float], optional): Диапазон случайных значений
            температуры `(min, max)`. По умолчанию (0.1, 1e7).

    Returns:
        float: Общее время выполнения предсказания в секундах.
    """
    rho_rand = np.random.uniform(rho_range[0], rho_range[1], n_queries)
    T_rand = np.random.uniform(T_range[0], T_range[1], n_queries)

    t0 = time.perf_counter()
    _ = model_predict_func(rho_rand, T_rand)
    t_total = time.perf_counter() - t0

    print(
        f"Скорость на {n_queries:,} запросах: {t_total:.3f} с "
        f"({n_queries / t_total:,.0f} запросов/сек)"
    )
    return t_total
