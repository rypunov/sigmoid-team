import torch
import math
from MLP import grad


def eos_loss_fn(model, rho, T, p_target, e_target, normalizer):
    """Физически-информированная функция потерь для уравнения состояния.

       Args:
           model (nn.Module): Нейросеть, принимающая (rho, T) и возвращающая
                          (P, E) в нормализованных единицах.
           rho (torch.Tensor): Нормализованные плотности формы ``(N, 1)``.
           T (torch.Tensor): Нормализованные температуры формы ``(N, 1)``.
           p_target (torch.Tensor): Целевые нормализованные давления ``(N, 1)``.
           e_target (torch.Tensor): Целевые нормализованные энергии ``(N, 1)``.
           normalizer: Объект нормализатора.

       Returns:
           dict[str, torch.Tensor]: Словарь с покомпонентными потерями:
               ``{"pde", "bc", "ic", "L_C", "L_th"}.
       """
    rho = rho.clone().requires_grad_(True)
    T = T.clone().requires_grad_(True)

    preds = model(torch.cat([rho, T], dim=1))
    p_pred = preds[:, 0:1]
    e_pred = preds[:, 1:2]

    dp_drho_n = grad(p_pred.sum(), rho)
    dp_dT_n = grad(p_pred.sum(), T)
    de_drho_n = grad(e_pred.sum(), rho)
    de_dT_n = grad(e_pred.sum(), T)

    sig_rho = normalizer.params["rho"]["std"]
    sig_T   = normalizer.params["T"]["std"]
    sig_P   = normalizer.params["P"]["std"]
    sig_E   = normalizer.params["E"]["std"]
    mu_rho  = normalizer.params["rho"]["mean"]
    mu_T    = normalizer.params["T"]["mean"]
    mu_P    = normalizer.params["P"]["mean"]

    ln10 = math.log(10)
    rho_phys = torch.pow(10.0, rho * sig_rho + mu_rho)
    T_phys   = T * sig_T + mu_T
    P_phys   = torch.pow(10.0, p_pred * sig_P + mu_P)

    T_dP_dT      = T_phys * (P_phys * ln10 * sig_P / sig_T) * dp_dT_n
    rho2_dE_drho = rho_phys * (sig_E / (ln10 * sig_rho)) * de_drho_n

    # [FIX] Huber-like loss для P: устойчивее к выбросам в log-пространстве
    res_p = p_pred - p_target
    delta = 0.1
    L_p = torch.mean(torch.where(torch.abs(res_p) < delta,
                                 0.5 * res_p ** 2,
                                 delta * (torch.abs(res_p) - 0.5 * delta)))

    L_e   = torch.mean((e_pred - e_target) ** 2)
    L_mod = torch.mean(torch.relu(-dp_drho_n) ** 2)
    L_C   = torch.mean(torch.relu(-de_dT_n) ** 2)

    residual_th = T_dP_dT + rho2_dE_drho - P_phys
    denom = torch.abs(T_dP_dT) + torch.abs(rho2_dE_drho) + torch.abs(P_phys) + 1e-6
    L_th = torch.mean((residual_th / denom) ** 2)

    return {"pde": L_p, "bc": L_e, "ic": L_mod, "L_C": L_C, "L_th": L_th}