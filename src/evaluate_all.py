import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np

from config import REPORT_DIR, MODEL_DIR
from data_loader import load_raw_regions, load_flat_dataset
from preprocessor import split_and_prepare
from baseline import BilinearEOS
from metrics import evaluate_model, benchmark_throughput


# --- Импорты ваших нейросетей ---
# from ML_Models.PINN_Al import PINN_Model
# from ML_Models.MLP import MLPModel
# from ML_Models.Vanilla import VanillaModel


def make_baseline_wrapper(normalizer, regions):
    """Обучает (если нужно) и возвращает обертку для бейзлайна."""
    model_path = MODEL_DIR / "baseline.pkl"

    if model_path.exists():
        print(f"   Загружаю бейзлайн из {model_path}")
        baseline = BilinearEOS.load(model_path)
    else:
        print(f"   {model_path} не найден — обучаю с нуля...")
        baseline = BilinearEOS()
        baseline.fit(regions)
        baseline.save(model_path)
        print(f"   Бейзлайн сохранен в {model_path}")

    def predict(rho_n, T_n):
        rho_real = np.power(
            10.0, rho_n * normalizer.params["rho"]["std"] + normalizer.params["rho"]["mean"]
        ) - 1e-12
        T_real = T_n * normalizer.params["T"]["std"] + normalizer.params["T"]["mean"]

        P_real, E_real = baseline.predict(rho_real, T_real)

        P_n = (np.log10(P_real + 1e-12) - normalizer.params["P"]["mean"]) / normalizer.params["P"]["std"]
        E_n = (E_real - normalizer.params["E"]["mean"]) / normalizer.params["E"]["std"]
        return P_n, E_n

    return predict


def make_nn_wrapper(normalizer, model_path, model_class, device="cpu"):
    """Обертка для нейросетей."""
    if not model_path.exists():
        print(f"⚠️ Веса {model_path} не найдены, пропускаем...")
        return None

    import torch

    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    def predict(rho_n, T_n):
        with torch.no_grad():
            x = torch.tensor(np.stack([rho_n, T_n], axis=1), dtype=torch.float32).to(device)
            P_n, E_n = model(x)
            return P_n.cpu().numpy(), E_n.cpu().numpy()

    return predict


def main():
    print("1. Загрузка данных...")
    ds = load_flat_dataset()
    regions = load_raw_regions()

    print("2. Подготовка (нормализация, сплит)...")
    prepared = split_and_prepare(ds)
    normalizer = prepared["normalizer"]
    test_data = prepared["test"]
    print(f"   Тестовых точек: {len(test_data['rho']):,}")

    models_to_eval = []

    # 1. Бейзлайн
    baseline_predict = make_baseline_wrapper(normalizer, regions)
    models_to_eval.append(("Bilinear Baseline", baseline_predict))

    # 2. Нейросети (раскомментируйте, когда будут готовы веса)
    # nn_path = MODEL_DIR / "pinn_weights.pth"
    # nn_predict = make_nn_wrapper(normalizer, nn_path, PINN_Model)
    # if nn_predict is not None:
    #     models_to_eval.append(("PINN", nn_predict))

    all_metrics = {}
    for name, predict_func in models_to_eval:
        metrics = evaluate_model(predict_func, test_data, normalizer, name=name)

        print(f"\nЗамер скорости для {name} (10k запросов, реальный диапазон)...")
        t_total = benchmark_throughput(
            predict_func,
            n_queries=10_000,
            rho_range=(test_data["rho"].min(), test_data["rho"].max()),
            T_range=(test_data["T"].min(), test_data["T"].max()),
        )
        metrics["time_10k_s"] = t_total

        all_metrics[name] = metrics

    report_path = REPORT_DIR / "all_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n✅ Метрики сохранены в {report_path}")


if __name__ == "__main__":
    main()