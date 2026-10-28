# ==========================================
# 4. Функция потерь (данные + термодинамика)
# ==========================================
import torch
import math
from MLP import grad


def eos_loss_fn(model, rho, T, p_target, e_target, normalizer):
    """
    Вычисляет 5 компонентов функции потерь для уравнения состояния.
    rho, T, p_target, e_target: нормализованные тензоры размера [batch_size, 1]
    normalizer: объект EOSNormalizer
    """
    rho.requires_grad_(True)
    T.requires_grad_(True)

    # 1. Прямой проход (склеиваем rho и T в один тензор [batch, 2])
    preds = model(torch.cat([rho, T], dim=1))
    p_pred = preds[:, 0:1]
    e_pred = preds[:, 1:2]

    # 2. Частные производные по НОРМАЛИЗОВАННЫМ переменным
    dp_drho_n = grad(p_pred.sum(), rho)
    dp_dT_n = grad(p_pred.sum(), T)
    de_drho_n = grad(e_pred.sum(), rho)
    de_dT_n = grad(e_pred.sum(), T)

    # 3. Достаем параметры нормализации (используем .params)
    sig_rho = normalizer.params["rho"]["std"]
    sig_T = normalizer.params["T"]["std"]
    sig_P = normalizer.params["P"]["std"]
    sig_E = normalizer.params["E"]["std"]

    mu_rho = normalizer.params["rho"]["mean"]
    mu_T = normalizer.params["T"]["mean"]
    mu_P = normalizer.params["P"]["mean"]

    # 4. Восстанавливаем физические переменные для уравнений
    # (rho и P были логарифмированы по основанию 10)
    ln10 = math.log(10)
    rho_phys = torch.pow(10.0, rho * sig_rho + mu_rho) - 1e-12
    T_phys = T * sig_T + mu_T
    P_phys = torch.pow(10.0, p_pred * sig_P + mu_P) - 1e-12

    # 5. Пересчет производных в физические величины (Chain Rule)
    # T * dP/dT = T * (P * ln(10) * sig_P / sig_T) * dp_dT_n
    T_dP_dT = T_phys * (P_phys * ln10 * sig_P / sig_T) * dp_dT_n

    # rho^2 * dE/drho = rho^2 * (sig_E / (ln(10) * rho * sig_rho)) * de_drho_n
    rho2_dE_drho = rho_phys * (sig_E / (ln10 * sig_rho)) * de_drho_n

    # dP/drho (для проверки стабильности)
    dP_drho_phys = P_phys * (sig_P / (ln10 * sig_rho * rho_phys)) * dp_drho_n

    # 6. Вычисление 5 слагаемых лосса
    # Ошибка по давлению (в нормализованном пространстве)
    L_p = torch.mean((p_pred - p_target) ** 2)

    # Ошибка по энергии (в нормализованном пространстве)
    L_e = torch.mean((e_pred - e_target) ** 2)

    # Механическая стабильность: max(0, -dP/drho)^2
    L_mod = torch.mean(torch.relu(-dP_drho_phys) ** 2)

    # Теплоемкость: max(0, -dE/dT)^2.
    # Т.к. sig_E, sig_T > 0, знак de_dT_n совпадает со знаком dE/dT
    L_C = torch.mean(torch.relu(-de_dT_n) ** 2)

    # Термодинамическая согласованность (Уравнение Максвелла)
    residual_th = T_dP_dT + rho2_dE_drho - P_phys
    L_th = torch.mean(residual_th ** 2) / (torch.mean(P_phys ** 2) + 1e-8)

    # Возвращаем словарь с ключами, которые ожидает Trainer (pde, bc, ic)
    return {
        "pde": L_p,  # L_p
        "bc": L_e,  # L_e
        "ic": L_mod,  # L_mod
        "L_C": L_C,
        "L_th": L_th
    }