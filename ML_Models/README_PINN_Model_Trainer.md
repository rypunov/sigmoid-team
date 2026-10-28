# PINN_Model_Trainer.py — Тренер и оценка PINN для EOS

## Назначение

Модуль содержит класс `PINN_Model_Trainer`, реализующий полный цикл работы с PINN-моделью:

1. Подготовку тензорных представлений train-данных на `device`.
2. Универсальный цикл обучения `train()` с поддержкой произвольной `loss_fn`, `scale` и `eval_fn`.
3. Физически-осмысленный `eval_fn` для оценки ошибки по `P` и `E`.
4. Специализированный `train_model()` для EOS-задачи.
5. Визуализацию истории обучения и распределений ошибок.

## Зависимости

- `torch`, `numpy`, `matplotlib`, `tqdm`
- `Loss.eos_loss_fn`

## Класс `PINN_Model_Trainer`

### Конструктор

```python
PINN_Model_Trainer(model, train_data, test_data, normalizer,
                   n_iters=5000, batch_size=2048)
```

| Аргумент | Описание |
|---|---|
| `model` | `nn.Module`, `[B,2] → [B,2]`. |
| `train_data` | `dict` с ключами `rho`, `T`, `P`, `E` (нормализованные numpy-массивы). |
| `test_data` | То же для теста. |
| `normalizer` | `EOSNormalizer` с `params` и методами `inverse_transform_P/E`. |
| `n_iters` | Число итераций `train_model`. |
| `batch_size` | Размер мини-батча. |

**В конструкторе** создаются тензоры на `device`:
- `self.rho_t`, `self.T_t`, `self.p_t`, `self.e_t` — формы `[N, 1]`.

### Методы

#### `train(loss_fn, n_iters, lr=1e-3, extra_params=(), eval_fn=None, log_every=500, desc="train", scale=...)`

Универсальный цикл обучения.

| Аргумент | Описание |
|---|---|
| `loss_fn` | Callable `() -> dict[str, Tensor]`. |
| `n_iters` | Количество итераций. |
| `lr` | Learning rate для Adam. |
| `extra_params` | Параметры, добавляемые к `model.parameters()` (например, обучаемые коэффициенты в `scale`). |
| `eval_fn` | Callable `(model) -> dict[str, float]` для метрик. |
| `log_every` | Частота логирования. |
| `scale` | Словарь весов для каждого ключа лосса. По умолчанию `pde=bc=1.0`, `ic=L_C=L_th=0.01`. |

**Схема шага:**

```python
terms = loss_fn()
for k in terms:
    terms[k] = terms[k] * scale[k]
loss = sum(terms.values())
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()  # ExponentialLR(gamma=0.9^(1/1000))
```

**Возвращает** `history: dict`:
- `"it"` — итерации;
- `"loss_<key>"` — взвешенные компоненты;
- `<key>` из `eval_fn` — метрики.

#### `rel_l2(pred, exact)`

`np.linalg.norm(pred−exact) / (norm(exact)+ε)` — относительная L2-ошибка.

#### `predict_eos(model, rho_np, T_np)` (`@torch.no_grad`)

Принимает нормализованные numpy-массивы, возвращает `(p_pred, e_pred)` в **нормализованных** единицах.

#### `eval_fn(m)`

Оценивает относительные L2-ошибки по **физическим** `P` и `E` (после `inverse_transform_*`).

#### `train_model(desc="EOS PINN")`

Специализированный цикл для EOS:

- Определяет `loss_fn()` — случайный батч индексов → `eos_loss_fn`.
- Вызывает `self.train(...)` с `eval_fn=self.eval_fn`.
- Печатает время обучения и финальные RE-L2.
- Строит `semilogy`-график истории лоссов.
- Возвращает `history`.

#### `visualization_train_res()`

Строит три subplot по `(log ρ_phys, log T_phys)`:

1. **P exact** — раскраска по `P_test_phys` (viridis).
2. **P relative error** — `|P_pred − P_true| / |P_true|` (Reds).
3. **E relative error** — `|E_pred − E_true| / |E_true|` (Blues).

## Формула лосса (в тренере)

```
L = 1.0·L_pde + 1.0·L_bc + 0.01·L_ic + 0.01·L_C + 0.01·L_th
```

Веса в `scale` выбраны эмпирически: `ic`, `L_C`, `L_th` — мягкие регуляризаторы, чтобы не доминировать над MSE по данным.

## Особенности реализации

- **Случайные батчи из фиксированного тензора** — быстро, но не гарантирует покрытие всех данных за «эпоху»; для точной эпохи использовать `torch.randperm`.
- **Scheduler шагает каждую итерацию** (не эпоху) — скорость затухания `gamma^(it/1000)`.
- **`eval_fn` вызывается только при логировании** (каждые 500 итераций), а не каждый шаг — экономия вычислений.
- **RE-L2 по P** считается в **физических** единицах — устойчивее к log-нормализации, чем MSE в нормализованном пространстве.

## Рекомендации

- Для стабильной сходимости `L_th` начинать с `scale['L_th'] = 1e-3` и увеличивать по мере уменьшения `L_pde`.
- Для диагностики «залипания» регуляризаторов смотреть `history['loss_L_C']` / `history['loss_L_th']` — они должны падать вместе с основными.
- При OOM уменьшить `batch_size`; граф автограда для 4 производных по 2 входам занимает ~5–8× памяти прямого прохода.
