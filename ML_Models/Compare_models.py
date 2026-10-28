"""
Сравнение PINN и Vanilla MLP на одинаковом тестовом сплите.

Использует:
  - config.py        : MODEL_DIR, REPORT_DIR
  - data_loader.py   : load_flat_dataset
  - preprocessor.py  : split_and_prepare, EOSNormalizer
  - metrics.py       : evaluate_model, benchmark_throughput  (общие метрики с baseline)
  - Loss.py          : eos_loss_fn  (для физических метрик PINN)
  - MLP.py           : MLP

Ожидаемые файлы моделей:
  ARTIFACTS_DIR/models/pinn_model.pt
  ARTIFACTS_DIR/models/vanilla_model.pt
  ARTIFACTS_DIR/models/normalization.json   (создаётся preprocessor.split_and_prepare)
"""
import sys
import json
from pathlib import Path
import pickle
# корень проекта в sys.path, как в eda.py / evaluate_all.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.config import MODEL_DIR, REPORT_DIR
from src.data_loader import load_flat_dataset
from src.preprocessor import split_and_prepare, EOSNormalizer
from src.metrics import evaluate_model
from MLP import MLP
from Loss import eos_loss_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ----------------------------------------------------------------------
# 1. Загрузка данных + сплит + нормализатор
# ----------------------------------------------------------------------
def prepare_data():
    ds = load_flat_dataset()
    prepared = split_and_prepare(ds)          # сохранит normalization.json в MODEL_DIR
    return prepared


# ----------------------------------------------------------------------
# 2. Загрузка весов (state_dict) для конкретной архитектуры
# ----------------------------------------------------------------------
def load_mlp(weights_path: Path) -> MLP:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Не найден файл весов: {weights_path}\n"
            f"Сначала обучите модель и сохраните state_dict в {MODEL_DIR}"
        )
    model = MLP(in_dim=2, out_dim=2, width=128, depth=5, activation=nn.GELU)
    state = torch.load(weights_path, map_location=DEVICE)
    # на случай, если сохранён чекпоинт-словарь {"model_state": ...}
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ----------------------------------------------------------------------
# 3. Обёртки predict(rho_n, T_n) -> (P_n, E_n) для metrics.evaluate_model
#    (совместимы с baseline из evaluate_all.py)
# ----------------------------------------------------------------------
def make_torch_predict(model: MLP):
    @torch.no_grad()
    def predict(rho_n: np.ndarray, T_n: np.ndarray):
        X = torch.tensor(
            np.stack([rho_n, T_n], axis=1), dtype=torch.float32
        ).to(DEVICE)
        out = model(X).cpu().numpy()
        return out[:, 0], out[:, 1]
    return predict


# ----------------------------------------------------------------------
# 4. Физические метрики PINN (PDE-остаток, нарушения монотонности)
#    Единица измерения — нормализованные величины, как в Loss.py
# ----------------------------------------------------------------------
@torch.no_grad()
def physics_metrics_pinn(model, rho_n, T_n, P_n, E_n, normalizer, n_phys=20_000):
    n = min(n_phys, len(rho_n))
    # eos_loss_fn требует автоград по входам — нельзя под @torch.no_grad
    pass


def physics_metrics_pinn_impl(model, rho_n, T_n, P_n, E_n, normalizer, n_phys=20_000):
    n = min(n_phys, len(rho_n))
    rho_t = torch.tensor(rho_n[:n], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    T_t   = torch.tensor(T_n[:n],   dtype=torch.float32, device=DEVICE).unsqueeze(1)
    p_t   = torch.tensor(P_n[:n],   dtype=torch.float32, device=DEVICE).unsqueeze(1)
    e_t   = torch.tensor(E_n[:n],   dtype=torch.float32, device=DEVICE).unsqueeze(1)

    terms = eos_loss_fn(model, rho_t, T_t, p_t, e_t, normalizer)
    return {f"phys_{k}": float(v.item()) for k, v in terms.items()}


# ----------------------------------------------------------------------
# 5. Основной сценарий сравнения
# ----------------------------------------------------------------------

print("\n1. Подготовка данных...")
prepared = prepare_data()

# --- normalizer: либо из pickle, либо пересчитать ---
# Вариант A: если сохраняли prepared.pkl
normalizer = EOSNormalizer.load(MODEL_DIR / "normalization.json")

test = prepared["test"]

rho_n = test["rho"]; T_n = test["T"]
P_n   = test["P"];   E_n = test["E"]
print(f"   test: {len(rho_n):,} точек")

# ---------- Vanilla ----------
print("\n2. Оценка Vanilla MLP...")
vanilla = load_mlp(MODEL_DIR / "vanilla_model.pt")
vanilla_pred = make_torch_predict(vanilla)
vanilla_metrics = evaluate_model(vanilla_pred, test, normalizer, name="Vanilla MLP")
vanilla_metrics["model"] = "Vanilla"

# ---------- PINN ----------
print("\n3. Оценка PINN...")
pinn = load_mlp(MODEL_DIR / "pinn_model1.pt")
pinn_pred = make_torch_predict(pinn)
pinn_metrics = evaluate_model(pinn_pred, test, normalizer, name="PINN EOS")
pinn_metrics["model"] = "PINN"

# физические метрики — только для PINN
print("\n4. Физические метрики PINN (PDE-residual, монотонность)...")
phys = physics_metrics_pinn_impl(pinn, rho_n, T_n, P_n, E_n, normalizer)
pinn_metrics.update(phys)

# ---------- Таблица ----------
keys_common = [
    ("P_MAPE_%",     "P MAPE, %"),
    ("P_median_%",   "P median, %"),
    ("P_p99_%",      "P p99, %"),
    ("E_MAPE_%",     "E MAPE, %"),
    ("E_median_%",   "E median, %"),
    ("E_p99_%",      "E p99, %"),
    ("P_RMSE",       "P RMSE"),
    ("E_RMSE",       "E RMSE"),
    ("Throughput_qps","Throughput, зап/с"),
]
header = f"{'метрика':<22}{'Vanilla':>16}{'PINN':>16}"
print("\n" + "=" * len(header))
print(header)
print("-" * len(header))
for k, label in keys_common:
    v = vanilla_metrics.get(k, float("nan"))
    p = pinn_metrics.get(k, float("nan"))
    print(f"{label:<22}{'—':>16}{p:>16.4g}")

print("-" * len(header))
for k, label in [
    ("phys_L_th", "PDE-остаток (L_th)"),
    ("phys_ic",   "наруш. dP/drho (L_mod)"),
    ("phys_L_C",  "наруш. dE/dT (L_C)"),
]:
    if k in pinn_metrics:
        print(f"{label:<22}{'—':>16}{pinn_metrics[k]:>16.4g}")
print("=" * len(header))

# ---------- Сохранение ----------
out = {
    "vanilla": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in vanilla_metrics.items()},
    "pinn":    {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in pinn_metrics.items()},
}
path = REPORT_DIR / "compare_metrics.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nМетрики сравнения сохранены: {path}")
