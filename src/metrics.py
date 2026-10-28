import numpy as np
import time


def calculate_errors(y_true: np.ndarray, y_pred: np.ndarray, is_energy: bool = False):
    if is_energy:
        error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1.0)
    else:
        error = np.abs(y_pred - y_true) / np.abs(y_true)
    return error


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def evaluate_model(model_predict_func, ds_test: dict, normalizer, name: str = "Model"):
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