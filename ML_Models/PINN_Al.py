import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from MLP import MLP, grad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print("Device:", device)


# ==========================================
# 3. Архитектура PINN для уравнения состояния
# ==========================================
class PINN_EOS(nn.Module):
    def __init__(self, model, data_scale):
        super().__init__()
        self.model = model
        self.rho_mean, self.rho_std = data_scale['rho_mean'], data_scale['rho_std']
        self.T_mean, self.T_std = data_scale['T_mean'], data_scale['T_std']
        self.P_scale = data_scale['P_scale']
        self.E_scale = data_scale['E_scale']

    def forward(self, rho_n, T_n):
        return self.model(torch.cat([rho_n, T_n], dim=1))

    def compute_thermo(self, rho_n, T_n):
        """Вычисляет P и E из свободной энергии Гельмгольца F(rho, T)."""
        rho_n = rho_n.clone().detach().requires_grad_(True)
        T_n = T_n.clone().detach().requires_grad_(True)

        F = self.forward(rho_n, T_n)

        # Автоматическое дифференцирование
        dF_dT_n = grad(F, T_n)
        dF_drho_n = grad(F, rho_n)

        # Преобразование обратно к физическим переменным через chain rule
        rho = torch.exp(rho_n * self.rho_std + self.rho_mean)
        T = torch.exp(T_n * self.T_std + self.T_mean)

        # Внутренняя энергия: E = F - T * dF/dT
        E_pred = F - T * dF_dT_n / self.T_std
        # Давление: P = rho^2 * dF/drho
        P_pred = (rho ** 2) * dF_drho_n / (rho * self.rho_std)

        return P_pred, E_pred


# ==========================================
# 4. Функция потерь (данные + термодинамика)
# ==========================================
def eos_loss_fn(model, rho, T, p_target, e_target, weights=None):
    """
    rho, T, p_target, e_target: тензоры размера [batch_size, 1]
    """
    if weights is None:
        # Веса для балансировки слагаемых (подбираются эмпирически)
        weights = {"L_e": 1.0, "L_p": 1.0, "L_mod": 1e-3, "L_C": 1e-3, "L_th": 1e-3}

    # Включаем автоградиент для входных переменных
    rho.requires_grad_(True)
    T.requires_grad_(True)

    # Прямой проход (p*, e*)
    preds = model(rho, T)
    p_pred = preds[:, 0:1]
    e_pred = preds[:, 1:2]

    # Вычисление частных производных
    dp_drho = grad(p_pred.sum(), rho)
    dp_dT = grad(p_pred.sum(), T)
    de_drho = grad(e_pred.sum(), rho)
    de_dT = grad(e_pred.sum(), T)

    # 1. Ошибка по энергии: L_e = sum((e* - e)^2) / sum(e^2)
    L_e = torch.sum((e_pred - e_target) ** 2) / (torch.sum(e_target ** 2) + 1e-8)

    # 2. Ошибка по давлению: L_p = sum((p* - p)^2) / sum(p^2)
    L_p = torch.sum((p_pred - p_target) ** 2) / (torch.sum(p_target ** 2) + 1e-8)

    # 3. Механическая стабильность: L_mod = sum(max(0, -dp*/drho)^2)
    L_mod = torch.sum(torch.relu(-dp_drho) ** 2)

    # 4. Положительная теплоемкость: L_C = sum(max(0, -de*/dT)^2)
    L_C = torch.sum(torch.relu(-de_dT) ** 2)

    # 5. Термодинамическая согласованность:
    # L_th = sum( (T * dp*/dT + rho^2 * de*/drho - p*)^2 ) / sum(p^2)
    residual_th = T * dp_dT + (rho ** 2) * de_drho - p_pred
    L_th = torch.sum(residual_th ** 2) / (torch.sum(p_target ** 2) + 1e-8)

    # Итоговая функция потерь
    total_loss = (weights["L_e"] * L_e + weights["L_p"] * L_p + weights["L_mod"] * L_mod +
                  weights["L_C"] * L_C + weights["L_th"] * L_th)

    return {"total": total_loss, "L_e": L_e, "L_p": L_p, "L_mod": L_mod, "L_C": L_C, "L_th": L_th}

# ==========================================
# 5. Тренер и цикл обучения
# ==========================================
class PINN_EOS_Trainer:
    def __init__(self, model_pinn, train_loader, val_loader, n_iters=5000):
        self.model_pinn = model_pinn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_iters = n_iters

    def train(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.model_pinn.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9 ** (1 / 1000))
        history = {"it": []}

        for epoch in tqdm(range(1, self.n_iters + 1), desc="Training PINN-EOS"):
            for batch in self.train_loader:
                rho_b, T_b, P_b, E_b = batch
                rho_b, T_b, P_b, E_b = rho_b.to(device), T_b.to(device), P_b.to(device), E_b.to(device)

                optimizer.zero_grad()
                loss_dict = eos_loss_fn(self.model_pinn, rho_b, T_b, P_b, E_b)
                loss_dict["total"].backward()
                optimizer.step()

            scheduler.step()

            if epoch % 200 == 0 or epoch == 1:
                # Валидация
                with torch.no_grad():
                    val_loss_P = 0.0
                    val_loss_E = 0.0
                    for batch_val in self.val_loader:
                        rho_val, T_val, P_val, E_val = [x.to(device) for x in batch_val]
                        P_vp, E_vp = self.model_pinn.compute_thermo(rho_val, T_val)
                        val_loss_P += torch.mean(((P_vp - P_val) / (P_val.abs() + 1e-6)) ** 2).item()
                        val_loss_E += torch.mean(((E_vp - E_val) / (E_val.abs() + 1e-6)) ** 2).item()

                history["it"].append(epoch)
                history.setdefault("train_total", []).append(loss_dict["total"].item())
                history.setdefault("val_relMSE_P", []).append(val_loss_P / len(self.val_loader))
                history.setdefault("val_relMSE_E", []).append(val_loss_E / len(self.val_loader))

        return history


# ==========================================
# 6. Запуск и визуализация
# ==========================================
if __name__ == "__main__":
    # Параметры масштабирования
    data_scale = {'rho_mean': float(np.mean(rho_n)), 'rho_std': float(np.std(rho_n)),
                  'T_mean': float(np.mean(T_n)), 'T_std': float(np.std(T_n)),
                  'P_scale': 1.0, 'E_scale': 1.0}

    base_nn = MLP(in_dim=2).to(device)
    pinn_model = PINN_EOS(base_nn, data_scale).to(device)
    trainer = PINN_EOS_Trainer(pinn_model, train_loader, val_loader, n_iters=5000)

    history = trainer.train(lr=1e-3)

    # Визуализация истории обучения
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["it"], history["val_relMSE_P"], label="Val Rel-MSE (P)")
    ax.semilogy(history["it"], history["val_relMSE_E"], label="Val Rel-MSE (E)")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Относительная MSE")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.title("Обучение PINN-EOS (термодинамически согласованный)")
    plt.tight_layout()
    plt.show()