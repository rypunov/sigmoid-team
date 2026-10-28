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
    """Создаёт обёртку предсказания для бейзлайновой модели BilinearEOS.

    Если сохранённый файл модели существует, загружает его; иначе обучает
    BilinearEOS на переданных регионах и сохраняет в MODEL_DIR / "baseline.pkl".
    Возвращаемая функция принимает нормализованные плотность и температуру,
    переводит их в физические величины, получает физические P и E от бейзлайна
    и возвращает их в нормализованном виде.
    
    Args:
        normalizer (EOSNormalizer): Объект нормализации с параметрами `params`
            для `rho`, `T`, `P` и `E`.
        regions (list[EOSRegion]): Список областей для обучения бейзлайна,
            если сохранённая модель отсутствует.
    
    Returns:
        callable: Функция `predict(rho_n, T_n) -> tuple[np.ndarray, np.ndarray]`,
            возвращающая нормализованные давление и энергию.
    """
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
        """Предсказывает нормализованные давление и энергию с помощью бейзлайна.

        Args:
            rho_n (np.ndarray): Нормализованная плотность, форма (N,).
            T_n (np.ndarray): Нормализованная температура, форма (N,).
        
        Returns:
            tuple[np.ndarray, np.ndarray]: Кортеж `(P_n, E_n)` с нормализованными
                давлением и энергией, каждый массив формы (N,).
        """
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
    """Создаёт обёртку предсказания для нейросетевой модели.
    
    Загружает веса модели из `model_path`, переводит модель в режим `eval`
    и на указанное устройство. Если файл весов не найден, возвращает `None`.
    
    Args:
        normalizer (EOSNormalizer): Объект нормализации (в текущей реализации
            не используется внутри обёртки, зарезервирован для совместимости).
        model_path (str | Path): Путь к файлу с весами модели.
        model_class (callable): Класс или фабрика, создающая экземпляр модели.
        device (str, optional): Устройство для вычислений (`"cpu"` или `"cuda"`).
            По умолчанию `"cpu"`.
    
    Returns:
        callable | None: Функция `predict(rho_n, T_n) -> tuple[np.ndarray, np.ndarray]`
            или `None`, если веса не найдены.
    """
    if not model_path.exists():
        print(f"⚠️ Веса {model_path} не найдены, пропускаем...")
        return None

    import torch

    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    def predict(rho_n, T_n):
        """Выполняет предсказание нейросетевой модели без вычисления градиентов.

        Args:
            rho_n (np.ndarray): Нормализованная плотность, форма (N,).
            T_n (np.ndarray): Нормализованная температура, форма (N,).
        
        Returns:
            tuple[np.ndarray, np.ndarray]: Кортеж `(P_n, E_n)` с нормализованными
                давлением и энергией, каждый массив формы (N,).
        """
        with torch.no_grad():
            x = torch.tensor(np.stack([rho_n, T_n], axis=1), dtype=torch.float32).to(device)
            P_n, E_n = model(x)
            return P_n.cpu().numpy(), E_n.cpu().numpy()

    return predict


def main():
    """Запускает оценку моделей уравнения состояния и сохраняет метрики.

    Выполняет загрузку данных, нормализацию и разбиение, формирует список
    моделей для оценки (бейзлайн и, при наличии весов, нейросети), вычисляет
    метрики качества и замеряет скорость предсказания на 10 000 запросов.
    Итоговые метрики сохраняются в REPORT_DIR / "all_metrics.json".
    
    Returns:
        None
    """
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
