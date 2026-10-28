# MLP.py — Полносвязная нейросеть и утилита автоградиента

## Назначение

Модуль содержит два компонента, используемых как «строительные блоки» во всех PINN-моделях:

1. **`MLP`** — конфигурируемая полносвязная сеть с Xavier-инициализацией.
2. **`grad(outputs, inputs)`** — обёртка над `torch.autograd.grad` с `create_graph=True`, необходимая для вычисления **вторых производных** (дифференцирование лосса по `ρ, T`).

## Зависимости

- `torch`, `torch.nn`
- `ML_Models.PINN_Al.device` — глобальный `torch.device` (CUDA/CPU).

> ⚠️ Импорт `device` из `PINN_Al` создаёт циклическую зависимость (`PINN_Al → MLP → PINN_Al`). В текущем коде это работает только потому, что `device` уже определён на уровне модуля до импорта `MLP`. **Рекомендуется** вынести `device` в отдельный модуль (`src.config`).

## Класс `MLP`

```python
MLP(in_dim=2, out_dim=2, width=128, depth=5, activation=nn.GELU)
```

| Аргумент | Описание |
|---|---|
| `in_dim` | Размерность входа (для EOS — 2: `[ρ_n, T_n]`). |
| `out_dim` | Размерность выхода (для EOS — 2: `[p_n, e_n]`). |
| `width` | Ширина скрытых слоёв. |
| `depth` | Число скрытых слоёв. |
| `activation` | Класс активации (по умолчанию `nn.GELU`). |

**Архитектура:**

```
Linear(in_dim, width) → activation → Linear(width, width) → ... (depth раз)
→ Linear(width, out_dim)
```

**Инициализация:**
- Веса — Xavier normal (`nn.init.xavier_normal_`).
- Смещения — нули.

### Метод `forward(xt)`

Принимает `[B, in_dim]`, возвращает `[B, out_dim]`.

## Функция `grad(outputs, inputs)`

```python
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True
    )[0]
```

**Почему именно так:**

| Флаг | Зачем нужен |
|---|---|
| `create_graph=True` | Позволяет дифференцировать результат по параметрам модели (для backprop через физические производные в лоссе). |
| `retain_graph=True` | Даёт возможность повторно вызывать `grad` по тому же вычислительному графу (4 производные в `Loss.py`). |
| `grad_outputs=ones_like` | Стандартный трюк для вычисления якобиана «поэлементно» как градиента суммы. Для скалярного `outputs` даст тот же результат, что и `outputs.sum()`. |

**Возвращает** тензор той же формы, что `inputs` — частную производную «на элемент».

## Пример использования

```python
from MLP import MLP, grad

model = MLP(in_dim=2, out_dim=2, width=128, depth=5).to(device)
x = torch.randn(64, 2, requires_grad=True)
y = model(x)

dp_drho = grad(y[:, 0:1].sum(), x)[:, 0:1]  # [64, 1]
```

## Возможные улучшения

- Заменить `torch.ones_like` на `torch.ones_like(outputs, requires_grad=False)` — экономит память.
- Добавить опциональные `weight_init` / `bias_init` для гибкости.
- Использовать `torch.func.jacrev` / `vmap` вместо множественных `autograd.grad` — часто в 2–3 раза быстрее.
