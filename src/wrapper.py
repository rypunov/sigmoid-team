"""
Модуль 4: Универсальная обертка для моделей EOS.
Обеспечивает единый интерфейс (predict, sound_speed, check_physics)
для Baseline, Vanilla и PINN.
"""
import numpy as np
import torch
from typing import Tuple, Dict


class EOSModelWrapper:
    """
    Обертка для любой модели EOS.
    Принимает физические единицы (г/см^3, K), возвращает физические единицы (ГПа, кДж/г).
    """

    def __init__(self, model, normalizer, model_type: str = "pytorch"):
        """
        Args:
            model: Обученная модель (PyTorch nn.Module, TorchScript или объект Baseline).
            normalizer: Объект EOSNormalizer (из preprocessor.py).
            model_type: Тип модели: "pytorch", "torchscript" или "baseline".
        """
        self.model = model
        self.normalizer = normalizer
        self.model_type = model_type.lower()

        # Для нейросетей переводим в режим инференса
        if self.model_type in ["pytorch", "torchscript"]:
            self.model.eval()

    def _to_norm(self, rho: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Переводит физические rho и T в нормированные (логарифмы + Z-score)."""
        p = self.normalizer.params
        rho_norm = (np.log10(rho + 1e-12) - p["rho"]["mean"]) / p["rho"]["std"]
        T_norm = (T - p["T"]["mean"]) / p["T"]["std"]
        return rho_norm, T_norm

    def predict(self, rho: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание давления и энергии.

        Args:
            rho: плотность [г/см^3], форма (N,)
            T: температура [K], форма (N,)

        Returns:
            P: давление [ГПа], форма (N,)
            E: энергия [кДж/г], форма (N,)
        """
        # Если это бейзлайн (интерполяция), он обычно принимает физические значения напрямую
        if self.model_type == "baseline":
            return self.model.predict(rho, T)

        # Для нейросетей
        rho_norm, T_norm = self._to_norm(rho, T)

        # Собираем входной тензор (N, 2)
        x = torch.stack([
            torch.tensor(rho_norm, dtype=torch.float32),
            torch.tensor(T_norm, dtype=torch.float32)
        ], dim=1)

        # Если модель на GPU, переносим тензор туда же
        if self.model_type == "pytorch":
            device = next(self.model.parameters()).device
            x = x.to(device)

        # Предсказание
        with torch.no_grad():
            if self.model_type == "pytorch":
                pred = self.model(x).cpu().numpy()
            elif self.model_type == "torchscript":
                pred = self.model(x).cpu().numpy()
            else:
                raise ValueError(f"Неизвестный тип модели: {self.model_type}")

        # Используем методы вашего EOSNormalizer для обратного пересчета
        P_phys = self.normalizer.inverse_transform_P(pred[:, 0])
        E_phys = self.normalizer.inverse_transform_E(pred[:, 1])

        return P_phys, E_phys

    def sound_speed(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Вычисляет скорость звука [км/с] через производные (autograd).

        Args:
            rho: плотность [г/см^3], форма (N,)
            T: температура [K], форма (N,)

        Returns:
            cs: скорость звука [км/с], форма (N,)
        """
        if self.model_type == "baseline":
            raise NotImplementedError("Скорость звука для Baseline не реализована через autograd.")

        rho_norm, T_norm = self._to_norm(rho, T)
        x = torch.stack([
            torch.tensor(rho_norm, dtype=torch.float32),
            torch.tensor(T_norm, dtype=torch.float32)
        ], dim=1).requires_grad_(True)

        device = next(self.model.parameters()).device
        x = x.to(device)

        y = self.model(x)
        P_norm, E_norm = y[:, 0], y[:, 1]

        # Считаем производные (autograd)
        dP = torch.autograd.grad(P_norm.sum(), x, retain_graph=True)[0]
        dE = torch.autograd.grad(E_norm.sum(), x)[0]

        p = self.normalizer.params

        # Переводим P_norm в физическое P для формулы цепного правила
        P_phys = self.normalizer.inverse_transform_P(P_norm.detach().cpu().numpy())
        rho_phys = rho

        # Правило цепочки для производных (с учетом логарифмов для P и rho)
        dP_drho = P_phys * (1.0 / (rho_phys * p["rho"]["std"])) * dP[:, 0].detach().cpu().numpy() * p["P"]["std"]
        dP_dT = P_phys * (1.0 / p["T"]["std"]) * dP[:, 1].detach().cpu().numpy() * p["P"]["std"]

        # Для энергии логарифма нет, поэтому проще
        dE_drho = (p["E"]["std"] / (rho_phys * p["rho"]["std"])) * dE[:, 0].detach().cpu().numpy()
        dE_dT = (p["E"]["std"] / p["T"]["std"]) * dE[:, 1].detach().cpu().numpy()

        # Формула скорости звука (в км/с)
        with np.errstate(divide="ignore", invalid="ignore"):
            cs2 = dP_drho + T / (rho_phys ** 2 * dE_dT) * dP_dT ** 2
            cs = np.sqrt(np.where(dE_dT > 0, cs2, np.nan))

        return cs

    def check_physics(self, rho: np.ndarray, T: np.ndarray) -> Dict[str, int]:
        """
        Проверяет физические ограничения модели.

        Args:
            rho: плотность [г/см^3]
            T: температура [K]

        Returns:
            dict: Флаги нарушений (количество точек с нарушениями).
        """
        P, E = self.predict(rho, T)

        violations = {
            "P_non_positive": int(np.sum(P <= 0)),
            "E_out_of_bounds": int(np.sum(E < -100)),  # Примерный минимум
            "total_violations": 0
        }

        # Если это нейросеть, проверим знаки производных
        if self.model_type != "baseline":
            rho_norm, T_norm = self._to_norm(rho, T)
            x = torch.stack([
                torch.tensor(rho_norm, dtype=torch.float32),
                torch.tensor(T_norm, dtype=torch.float32)
            ], dim=1).requires_grad_(True)

            device = next(self.model.parameters()).device
            x = x.to(device)
            y = self.model(x)

            dP = torch.autograd.grad(y[:, 0].sum(), x, retain_graph=True)[0]
            dE = torch.autograd.grad(y[:, 1].sum(), x)[0]

            # Знаки производных по нормированным осям совпадают с физическими
            violations["dP_drho_negative"] = int(torch.sum(dP[:, 0] < 0).item())
            violations["dE_dT_negative"] = int(torch.sum(dE[:, 1] < 0).item())

        # Суммируем все нарушения (кроме самого total_violations)
        violations["total_violations"] = sum([v for k, v in violations.items() if k != "total_violations"])
        return violations