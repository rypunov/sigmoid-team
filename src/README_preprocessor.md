# preprocessor.py — Нормализация данных и разбиение на train/test

## Назначение

Модуль реализует **предобработку** табличных данных EOS:

- `EOSNormalizer` — нормализация/денормализация `ρ, T, P, E` с сохранением параметров в JSON.
- `split_and_prepare` — разбиение 80/20, фит нормализатора **только на train**, применение ко всему датасету, сохранение `normalization.json`.

## Зависимости

- `json`, `numpy`, `pathlib.Path`
- `sklearn.model_selection.train_test_split`
- `src.config.MODEL_DIR`

## Класс `EOSNormalizer`

### Схема нормализации

| Величина | Преобразование | Причина |
|---|---|---|
| `ρ` | `log10(ρ + 1e-12)`, затем Z-score | Диапазон `ρ` — многие порядки. |
| `P` | `log10(P + 1e-12)`, затем Z-score | Диапазон `P` — до 12 порядков. |
| `T` | Z-score | Умеренный диапазон. |
| `E` | Z-score | Может быть отрицательной; log не применим. |

**Параметры** хранятся в `self.params`:

```python
{
  "rho": {"mean": ..., "std": ..., "use_log": True},
  "T":   {"mean": ..., "std": ..., "use_log": False},
  "P":   {"mean": ..., "std": ..., "use_log": True},
  "E":   {"mean": ..., "std": ..., "use_log": False},
}
```

### Метод `fit(rho, T, P, E)`

Считает `mean` и `std` на **тренировочных** данных.

- Для `ρ` и `P` — по `log10(x + 1e-12)`.
- `std` — популяционное СКО (`np.std`, `ddof=0`).
- Возвращает `self` (для chaining).

### Метод `transform(rho, T, P, E)`

Применяет нормализацию:

```python
rho_norm = (log10(rho + 1e-12) − μ_ρ) / σ_ρ
T_norm   = (T − μ_T) / σ_T
P_norm   = (log10(P + 1e-12) − μ_P) / σ_P
E_norm   = (E − μ_E) / σ_E
```

Возвращает кортеж `(rho_norm, T_norm, P_norm, E_norm)`.

### `inverse_transform_P(P_norm)`

```python
log_P = P_norm * σ_P + μ_P
return 10**log_P − 1e-12
```

### `inverse_transform_E(E_norm)`

```python
return E_norm * σ_E + μ_E
```

### `save(path=MODEL_DIR / "normalization.json")`

Сохраняет `self.params` в JSON (`indent=4`, `ensure_ascii=False`). Печатает путь.

### `load(path=MODEL_DIR / "normalization.json")` (`@classmethod`)

Загружает `params` из JSON и возвращает новый `EOSNormalizer`.

## Функция `split_and_prepare(ds, random_state=42)`

**Вход:** `ds` — словарь из `load_flat_dataset()` с ключами `rho, T, P, E`.

**Шаги:**

1. `indices = np.arange(len(ds["rho"]))`.
2. `train_test_split(indices, test_size=0.2, random_state=42)` — **случайное** разбиение (не стратифицированное).
3. `normalizer.fit(...)` — **только на train** (важно: без утечки теста).
4. `normalizer.transform(...)` — на **всём** датасете.
5. `normalizer.save()` — сохраняет JSON в `MODEL_DIR`.
6. Возвращает:

```python
{
  "train": {"rho": ..., "T": ..., "P": ..., "E": ...},
  "test":  {"rho": ..., "T": ..., "P": ..., "E": ...},
  "normalizer": normalizer
}
```

**Важно:** `random_state=42` фиксирован для воспроизводимости. Если нужен другой сплит — передать параметр.

## Пример использования

```python
from data_loader import load_flat_dataset
from preprocessor import split_and_prepare

ds = load_flat_dataset()
prepared = split_and_prepare(ds)

train = prepared["train"]
test = prepared["test"]
normalizer = prepared["normalizer"]

P_real = normalizer.inverse_transform_P(test["P"])
```

## Особенности и подводные камни

- **`+1e-12`** защищает от `log10(0)`, но **не** от отрицательных `ρ` или `P`. Если в данных есть `P < 0`, `log10` даст `nan` — предварительно валидировать.
- **`std` может быть 0** (если все значения константны) — деление на ноль. Стоит добавить `max(std, 1e-12)`.
- **`E` может быть отрицательной** — Z-score это допускает, но `inverse_transform_E` вернёт отрицательное значение (это нормально для энергии).
- **`random_state=42`** — фиксирован; для кросс-валидации нужно вызывать `split_and_prepare` с разными seed.
- **Сохранение при каждом вызове** — `split_and_prepare` перезаписывает `normalization.json`; при параллельных запусках возможна гонка.
- **Нет `inverse_transform_rho` / `inverse_transform_T`** — при необходимости восстановить физические `ρ, T` (например, для визуализации) формулу нужно писать вручную (как в `PINN_Model_Trainer.visualization_train_res`).

## Рекомендации

- Добавить `inverse_transform_rho` и `inverse_transform_T` для симметрии.
- Валидировать входы (`np.isfinite`, `rho > 0`, `P > 0`) в `fit` и `transform`.
- Использовать `ddof=1` для `std`, если данные — выборка (сейчас `ddof=0`).
- Для стратификации по `region` — передавать `stratify=ds["region"]` в `train_test_split` (требует доработки API).
