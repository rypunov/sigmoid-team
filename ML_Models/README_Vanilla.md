# Vanilla.py — Baseline: обычная (нефизическая) нейросеть для EOS

## Назначение

Скрипт обучает **чисто data-driven** MLP на табличных данных уравнения состояния **без физических регуляризаторов**. Используется как baseline для сравнения с PINN по точности, скорости инференса, размеру и нарушениям физики.

## Зависимости

- `torch`, `numpy`, `matplotlib`, `json`, `os`, `time`
- `src.config.MODEL_FILE`, `src.data_loader`, `src.preprocessor`, `MLP`

## Гиперпараметры

```python
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 5000
BATCH_SIZE = 4096
LR         = 1e-3
```

## Основные функции

### `train_model(model, X_train, Y_train, X_test, Y_test)`

Обычный SGD-цикл с `MSELoss`:

- Adam + `ExponentialLR(gamma=0.9^(1/1000))`.
- Перемешивание через `torch.randperm` — **истинная эпоха** (в отличие от PINN-тренера).
- Логирование каждые 100 эпох.
- Сохраняет `state_dict` в `MODEL_FILE`.

**Возвращает** `(train_losses, test_losses)` — списки средних MSE.

### `compute_derivatives(model, X, n_points=5)`

Считает `[∂P/∂ρ, ∂P/∂T, ∂E/∂ρ, ∂E/∂T]` в **нормализованных** осях для `n_points` случайных точек теста (через `torch.autograd.grad`).

### `print_derivatives(points, grads)`

Печатает значения в человекочитаемом виде.

### `plot_learning_curve(train_losses, test_losses)`

`semilogy` train/test MSE → `learning_curve_vanilla.png`.

### `plot_error_heatmap(model, X_test, Y_test, normalizer)`

Два scatter-plot'а по `(log10 ρ, T)`:
- ошибка предсказания `P` (норм.);
- ошибка предсказания `E` (норм.).

→ `error_heatmap_vanilla.png`.

### `plot_violation_map(model, X_test, normalizer)`

Проверяет **физические нарушения** на тесте:

| Нарушение | Условие |
|---|---|
| Отрицательное давление | `P_phys ≤ 0` |
| Неустойчивость по плотности | `dP/dρ < 0` |
| Отрицательная теплоёмкость | `dE/dT < 0` |

Строит карту `(log10 ρ, T)`: серые точки — OK, красные — нарушения.  
→ `violation_map_vanilla.png`.

**Печатает** абсолютное число и долю нарушений в %.

### `evaluate_vanilla(model, X_test, Y_test, normalizer, n_speed=100_000, out_path="vanilla_metrics.json")`

Считает метрики для итоговой таблицы:

| Метрика | Описание |
|---|---|
| `MAPE_P_percent` | Средняя абсолютная процентная ошибка `P`. |
| `MAPE_E_percent` | Средняя абсолютная процентная ошибка `E`. |
| `R2_P` | R² **в лог-шкале** (P меняется на 12 порядков). |
| `size_mb` | Размер `MODEL_FILE` на диске. |
| `speed_queries_per_sec` | Пропускная способность на `100 000` точках (с синхронизацией CUDA). |

Результат сохраняется в `vanilla_metrics.json` и возвращается как `dict`.

### `save_to_torchscript(model, device, out_path="model_scripted.pt")`

Трейсит модель `torch.jit.trace` на `dummy_input = torch.randn(1, 2)` и сохраняет. Печатает размер в МБ.

## Сценарий `__main__`

1. Устанавливает `SEED` для `torch` и `numpy`.
2. Загружает данные (`load_flat_dataset`), нормализует (`split_and_prepare`).
3. Создаёт `MLP(2, 2, width=128, depth=5, activation=nn.GELU)`.
4. `train_model(...)` — обучение.
5. Печатает 5 случайных точек с производными.
6. `plot_learning_curve` — кривая.
7. `plot_error_heatmap` — тепловые карты ошибок.
8. `plot_violation_map` — карта физических нарушений.
9. `evaluate_vanilla` — метрики → JSON.
10. `save_to_torchscript` — экспорт в `vanilla_scripted.pt`.

## Что важно знать

- **Vanilla не знает о физике** — на `dP/dρ < 0` и `dE/dT < 0` часто ловит нарушения (это и есть цель сравнения).
- **R² по P в лог-шкале** — потому что в физической шкале `P` меняется на 12 порядков, и обычный R² будет ≈ 1 у любой модели.
- **Скорость инференса** измеряется с `torch.cuda.synchronize()` до/после — иначе цифры занижены из-за асинхронности CUDA.
- **TorchScript** даёт независимый от Python артефакт, пригодный для деплоя (C++/mobile).

## Сравнение с PINN (`PINN_Al.py`)

| Характеристика | Vanilla | PINN |
|---|---|---|
| Лосс | Только MSE | MSE + 3 физических регуляризатора |
| Нарушения `dP/dρ<0`, `dE/dT<0` | Обычно есть | Подавлены штрафами |
| Время обучения | ~1× | ~2–4× (автоград вторых производных) |
| Размер | такой же | такой же |
| Скорость инференса | такая же | такая же |

**Ожидание:** Vanilla даёт меньший MSE на train/test (нет конфликта с физикой), но чаще нарушает устойчивость. PINN чуть проигрывает по MSE, но выигрывает по физической корректности.

## Известные ограничения

- Скрипт делает много `.png` без указания пути — файлы падают в CWD.
- `evaluate_vanilla` открывает `MODEL_FILE` для `os.path.getsize`; если файл не сохранён — упадёт.
- `save_to_torchscript` использует `torch.jit.trace`, чувствительный к data-dependent control flow (в текущем MLP его нет — OK).
