# metrics.py — Метрики качества и бенчмарк пропускной способности

## Назначение

Модуль содержит функции для **единообразной оценки** моделей EOS:

- относительные ошибки (`MAPE`, медиана, p99);
- `RMSE`, `NRMSE`, `R²` (обычный и в лог-шкале);
- время инференса и пропускную способность (запросов/сек).

Используется в `evaluate_all.py` и `Vanilla.py`.

## Зависимости

- `numpy`, `time`

## Функции

### `calculate_errors(y_true, y_pred, is_energy=False)`

Считает **поэлементную относительную ошибку**.

| Случай | Формула |
|---|---|
| `is_energy=False` (P) | `|y_pred − y_true| / |y_true|` |
| `is_energy=True` (E) | `|y_pred − y_true| / (|y_true| + 1.0)` |

**Почему `+1.0` для энергии:** `E` может быть близка к нулю или менять знак; добавка стабилизирует знаменатель. Для `P` подразумевается `P > 0` (физически), поэтому `+1.0` не нужен.

**Возвращает:** массив той же формы, что входы.

### `calculate_r2(y_true, y_pred)`

Классический коэффициент детерминации:

```
R² = 1 − Σ(y_true − y_pred)² / Σ(y_true − mean(y_true))²
```

- Если `ss_tot == 0` (константа) — возвращает `nan`.

### `evaluate_model(model_predict_func, ds_test, normalizer, name="Model")`

Главная функция оценки.

**Вход:**

| Аргумент | Описание |
|---|---|
| `model_predict_func` | `(rho_n, T_n) -> (P_n, E_n)` в **нормализованном** пространстве. |
| `ds_test` | `dict` с ключами `rho`, `T`, `P`, `E` (нормализованные numpy-массивы). |
| `normalizer` | `EOSNormalizer` с `inverse_transform_P/E`. |
| `name` | Имя модели для печати и отчёта. |

**Шаги:**

1. Замеряет время `model_predict_func` (включая forward, без учёта `inverse_transform`).
2. Восстанавливает **физические** `P_pred`, `E_pred`, `P_true`, `E_true` через `normalizer`.
3. Считает ошибки:
   - `P_err = calculate_errors(P_true, P_pred, is_energy=False)`,
   - `E_err = calculate_errors(E_true, E_pred, is_energy=True)`.
4. Считает `RMSE`, `NRMSE` (нормировка на `max − min`), `R²`, `R²` в лог-шкале:
   - `P_R2_log = calculate_r2(log1p(|P_true|), log1p(|P_pred|))`,
   - `E_R2_log = calculate_r2(sign(E)·log1p(|E|), sign(E_pred)·log1p(|E_pred|))`.
5. `Throughput_qps = len(rho_n) / t_pred`.

**Возвращает** словарь:

| Ключ | Описание |
|---|---|
| `model_name` | Имя модели. |
| `P_MAPE_%`, `P_median_%`, `P_p99_%` | Средняя, медианная, 99-я перцентиль относительной ошибки `P`. |
| `P_RMSE`, `P_NRMSE`, `P_R2`, `P_R2_log` | Метрики `P`. |
| `E_*` | Аналогично для `E`. |
| `Inference_time_s` | Время инференса на тесте (сек). |
| `Throughput_qps` | Запросов в секунду. |

**Печатает** компактный отчёт в stdout.

### `benchmark_throughput(model_predict_func, n_queries=100_000, rho_range=(0.001, 10000), T_range=(0.1, 1e7))`

Генерирует `n_queries` **равномерно распределённых** точек в заданных диапазонах `(rho_n, T_n)` и замеряет суммарное время вызова.

**Возвращает:** `t_total` (секунды).

**Печатает:** `Скорость на N запросах: T с (Q запросов/сек)`.

**Важно:** диапазоны по умолчанию — **глобальные** (`rho ∈ [1e-3, 1e4]`), а не реальные. В `evaluate_all.py` передаются реальные границы теста.

## Особенности

- **`P_R2` в физической шкале** почти всегда близок к 1 из-за огромного диапазона `P` (12 порядков) — поэтому дополнительно считается `P_R2_log`.
- **`NRMSE`** нормируется на `max − min`, а не на `mean` — устойчивее к сдвигу.
- **`Throughput_qps`** — «сырое» число; для честного сравнения с GPU-моделями нужен `torch.cuda.synchronize()` (в `Vanilla.py` это делается, здесь — нет).
- **`calculate_errors`** для `P` не защищает от `y_true == 0` — при `P = 0` получится `inf`/`nan`. Для EOS `P > 0`, но при экстраполяции бейзлайна возможно `P < 0` или `0`.

## Пример использования

```python
from metrics import evaluate_model, benchmark_throughput

metrics = evaluate_model(predict_func, test_data, normalizer, name="PINN")
t = benchmark_throughput(predict_func, n_queries=10_000,
                         rho_range=(test_data["rho"].min(), test_data["rho"].max()),
                         T_range=(test_data["T"].min(), test_data["T"].max()))
```

## Рекомендации

- Добавить `np.errstate(divide='ignore', invalid='ignore')` в `calculate_errors`.
- Считать `MAPE` только по точкам с `|y_true| > ε` (иначе заниженная/завышенная оценка).
- Для GPU-моделей добавить опциональный `sync_fn` в `benchmark_throughput`.
- Возвращать `t_pred` и `qps` в `benchmark_throughput` для единообразия.
