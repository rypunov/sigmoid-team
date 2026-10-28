# evaluate_all.py — Сравнительная оценка всех моделей EOS

## Назначение

Скрипт запускает **единый бенчмарк** для всех моделей (бейзлайн + нейросети), считает метрики через `metrics.evaluate_model`, замеряет пропускную способность и сохраняет результаты в `REPORT_DIR / "all_metrics.json"`.

## Зависимости

- `json`, `numpy`
- `config.REPORT_DIR`, `config.MODEL_DIR`
- `data_loader.load_flat_dataset`, `data_loader.load_raw_regions`
- `preprocessor.split_and_prepare`
- `baseline.BilinearEOS`
- `metrics.evaluate_model`, `metrics.benchmark_throughput`
- (опционально) нейросетевые модели — импорты закомментированы.

## Функции

### `make_baseline_wrapper(normalizer, regions)`

Обучает (если нужно) `BilinearEOS` и возвращает **функцию-обёртку** `predict(rho_n, T_n)`, работающую в **нормализованном** пространстве.

**Логика:**

1. Если `MODEL_DIR / "baseline.pkl"` существует — загружает.
2. Иначе — обучает на `regions` и сохраняет.
3. Обёртка:
   - `rho_real = 10^(rho_n · σ_ρ + μ_ρ) − 1e-12`,
   - `T_real = T_n · σ_T + μ_T`,
   - `P_real, E_real = baseline.predict(rho_real, T_real)`,
   - `P_n = (log10(P_real + 1e-12) − μ_P) / σ_P`,
   - `E_n = (E_real − μ_E) / σ_E`.
4. Возвращает `(P_n, E_n)`.

**Зачем:** привести выходы бейзлайна к тому же нормализованному пространству, в котором работают нейросети, чтобы `evaluate_model` мог единообразно вызывать `inverse_transform_*`.

### `make_nn_wrapper(normalizer, model_path, model_class, device="cpu")`

Универсальная обёртка для нейросетей.

**Логика:**

1. Если `model_path` не существует — печатает предупреждение и возвращает `None`.
2. Создаёт `model_class()`, загружает `state_dict`, переводит в `eval()` и на `device`.
3. Обёртка `predict(rho_n, T_n)`:
   - собирает `torch.tensor(np.stack([rho_n, T_n], axis=1))`,
   - делает forward без градиентов,
   - возвращает `(P_n, E_n)` как numpy-массивы.

**Ожидается**, что `model_class` — это `nn.Module`, возвращающий кортеж `(P_n, E_n)`.

### `main()`

**Пошаговый сценарий:**

1. `load_flat_dataset()` — плоские данные.
2. `load_raw_regions()` — регионы для бейзлайна.
3. `split_and_prepare(ds)` — train/test + `normalizer`.
4. Печатает число тестовых точек.
5. Создаёт список `models_to_eval`:
   - `("Bilinear Baseline", baseline_predict)` — всегда.
   - Нейросети — **закомментированы**; при раскомментировании добавятся `PINN`, `MLP`, `Vanilla`.
6. Для каждой модели:
   - `evaluate_model(...)` — печатает и возвращает словарь метрик.
   - `benchmark_throughput(..., n_queries=10_000, rho_range=..., T_range=...)` — замер на **реальном диапазоне** тестовых данных (не глобальном).
   - Добавляет `time_10k_s` в метрики.
7. Сохраняет `all_metrics.json` в `REPORT_DIR`.

## Пример вывода

```
=== Bilinear Baseline ===
P: MAPE = 2.1345%, Median = 0.9876%, p99 = 12.3456%, RMSE = 1.23e-02, NRMSE = 3.45%, R² = 0.999123 (log R² = 0.998765)
E: MAPE = 1.2345%, Median = 0.5432%, p99 = 8.7654%, RMSE = 2.34e-03, NRMSE = 1.23%, R² = 0.999876 (log R² = 0.999543)
Скорость: 12,345 запросов/сек (время: 0.81 с)
Скорость на 10,000 запросах: 0.85 с (11,765 запросов/сек)
```

## Особенности

- **Единый интерфейс:** все модели должны предоставлять `predict(rho_n, T_n) -> (P_n, E_n)` в нормализованном пространстве.
- **Бенчмарк на реальном диапазоне:** `rho_range` и `T_range` берутся из теста — это честнее, чем глобальный `(0.001, 10000)`.
- **Сохранение:** `json.dump` с `indent=4`, кодировка UTF-8.

## Как добавить модель

1. Импортировать класс модели (раскомментировать в шапке).
2. В `main()` добавить:

```python
nn_path = MODEL_DIR / "pinn_weights.pth"
nn_predict = make_nn_wrapper(normalizer, nn_path, PINN_Model)
if nn_predict is not None:
    models_to_eval.append(("PINN", nn_predict))
```

## Ограничения

- **`make_nn_wrapper`** ожидает, что `model_class()` создаётся без аргументов. Если модель требует `width/depth` — передавать `lambda: MLP(2, 2, width=128, depth=5)`.
- **`benchmark_throughput`** использует **равномерное** распределение в диапазоне, а не реальное распределение точек — скорость может отличаться от продакшн-сценария.
- **Нет сравнения с PINN по умолчанию** — нужно вручную раскомментировать и указать путь к весам.
- **`time_10k_s`** — это время **всего** бенчмарка на 10k точках, а не на одну; в отчёте может путать.

## Рекомендации

- Добавить `argparse` для выбора моделей и `n_queries`.
- Логировать версии библиотек и `git commit` в отчёт.
- Сохранять не только JSON, но и markdown-таблицу для отчёта.
