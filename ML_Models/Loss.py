import torch
from MLP import grad


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
