# PINN_Al.py — Точка входа для обучения PINN-модели уравнения состояния

## Назначение

Главный скрипт запуска обучения **физически-информированной нейросети** для уравнения состояния. Содержит:

- Загрузку и предобработку данных (`load_flat_dataset`, `split_and_prepare`).
- Инициализацию модели `MLP`.
- Настройку и запуск `PINN_Model_Trainer`.
- Визуализацию результатов после обучения.

## Зависимости

- `torch`, `numpy`, `matplotlib`, `tqdm`
- `MLP`, `src.data_loader`, `src.preprocessor`, `PINN_Model_Trainer`

## Структура модуля

### Класс `PINN_EOS` (обёртка, дублирует функциональность MLP)

```python
class PINN_EOS(nn.Module):
    def __init__(self, model, data_scale):
        ...
    def forward(self, rho_n, T_n):
        return self.model(torch.cat([rho_n, T_n], dim=1))
```

**Назначение:** хранить масштабы нормализации (`rho_mean`, `rho_std`, `T_mean`, `T_std`, `P_scale`, `E_scale`) вместе с сетью.

> ⚠️ В текущем скрипте класс **не используется** — тренер работает напрямую с `MLP` и объектом `normalizer`. Класс — наследие предыдущих версий.

### Основной сценарий запуска

```python
print("Загрузка данных...")
ds = load_flat_dataset()

print("Разбиение и нормализация...")
prepared = split_and_prepare(ds)

train_data = prepared["train"]
test_data = prepared["test"]
normalizer = prepared["normalizer"]

model = MLP(in_dim=2, out_dim=2).to(device)

trainer = PINN_Model_Trainer(
    model=model, train_data=train_data, test_data=test_data,
    normalizer=normalizer, n_iters=10_000, batch_size=4096
)

history = trainer.train_model(desc="PINN EOS")
trainer.visualization_train_res()
```

## Параметры обучения

| Параметр | Значение по умолчанию | Комментарий |
|---|---|---|
| `n_iters` | `10_000` | Число итераций SGD (не эпох!). |
| `batch_size` | `4096` | Размер мини-батча, выбираемого случайно. |
| `lr` | `1e-3` (внутри тренера) | Adam + `ExponentialLR(gamma=0.9^(1/1000))`. |
| `torch.manual_seed` | `0` | Воспроизводимость. |
| `np.random.seed` | `0` | Воспроизводимость разбиения. |

## Что происходит при запуске

1. Определяется `device` (`cuda` при наличии).
2. Загружается плоский датасет (`load_flat_dataset`).
3. Данные делятся на train/test, вычисляются параметры нормализации (`split_and_prepare`).
4. Создаётся `MLP(2→2)` со 128 нейронами и 5 скрытыми слоями.
5. Создаётся тренер; на каждой итерации:
   - случайный батч → `eos_loss_fn`;
   - суммарный лосс → `Adam.step()` → `scheduler.step()`;
   - каждые 500 итераций логируются компоненты и метрики ошибки `P/E`.
6. После обучения:
   - выводятся RE-L2 ошибки по `P` и `E` на тесте;
   - строится `semilogy`-график истории лоссов;
   - вызывается `visualization_train_res()` (карты `P`, `P-err`, `E-err` по `(log ρ, log T)`).

## Логирование

Метод `train()` тренера пишет в `history`:
- `it` — номера итераций;
- `loss_pde`, `loss_bc`, `loss_ic`, `loss_L_C`, `loss_L_th`;
- `error_P`, `error_E` (из `eval_fn`).

## Ожидаемые результаты

Для качественного обучения EOS:
- `RE-L2 P` ~ 1e-3…1e-2 (P меняется в log-шкале, поэтому даже хорошие модели дают заметные RE-L2);
- `RE-L2 E` ~ 1e-3…1e-2;
- Лоссы `L_th`, `L_C`, `ic` должны **монотонно убывать** на графике; их «полочки» означают насыщение регуляризаторов.

## Известные ограничения

- `n_iters=10_000` может быть недостаточно для сходимости термодинамических регуляризаторов — при проблемах увеличивать до 30–50k.
- Скрипт **не сохраняет** веса. Для персистентности добавить `torch.save(model.state_dict(), ...)` после обучения.
- Отсутствует `if __name__ == "__main__":` — обучение запускается при импорте модуля. Оборачивайте, если планируете использовать как библиотеку.
