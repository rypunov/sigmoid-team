"""
Обучение Vanilla 
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import os
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.config import MODEL_FILE
from src.data_loader import load_flat_dataset
from src.preprocessor import split_and_prepare
from MLP import MLP

# ПАРАМЕТРЫ ОБУЧЕНИЯ
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 5000
BATCH_SIZE = 4096
LR = 1e-3

# ОБУЧЕНИЕ
def train_model(model, X_train, Y_train, X_test, Y_test):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9 ** (1 / 1000))

    n = len(X_train)
    train_losses, test_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0

        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            loss = loss_fn(model(X_train[idx]), Y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            total += loss.item() * len(idx)

        train_losses.append(total / n)

        model.eval()
        with torch.no_grad():
            test_losses.append(loss_fn(model(X_test), Y_test).item())

        if epoch == 1 or epoch % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Эпоха {epoch:5d}/{EPOCHS} | train MSE={train_losses[-1]:.3e} | test MSE={test_losses[-1]:.3e} | LR={current_lr:.2e}")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Веса сохранены в {MODEL_FILE}")
    return train_losses, test_losses

# ПРОИЗВОДНЫЕ
def compute_derivatives(model, X, n_points=5):
    model.eval()
    idx = torch.randperm(len(X), device=DEVICE)[:n_points]
    x = X[idx].clone().requires_grad_(True)
    y = model(x)

    grads = []
    for k in range(2): # k=0 -> P, k=1 -> e
        g = torch.autograd.grad(y[:, k].sum(), x, retain_graph=True)[0]
        grads.append(g)
    grads = torch.stack(grads, dim=1)
    return x.detach(), grads.detach()

def print_derivatives(points, grads):
    print("\n Частные производные (по нормированным осям) ")
    for i in range(len(points)):
        r, t = points[i].cpu().numpy()
        g = grads[i].cpu().numpy()
        print(f"Точка {i + 1}: rho_norm={r:.4f}, T_norm={t:.4f}")
        print(f"   dP/drho_norm = {g[0, 0]: .5f}   dP/dT_norm = {g[0, 1]: .5f}")
        print(f"   dE/drho_norm = {g[1, 0]: .5f}   dE/dT_norm = {g[1, 1]: .5f}")

# ВИЗУАЛИЗАЦИЯ
def plot_learning_curve(train_losses, test_losses):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test")
    plt.yscale("log")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("Кривая обучения")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve_vanilla.png", dpi=150)
    plt.show()

def plot_error_heatmap(model, X_test, Y_test, normalizer):
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
    err = (pred - Y_test).abs().cpu().numpy()

    x_np = X_test.cpu().numpy()
    log_rho = x_np[:, 0] * normalizer.params["rho"]["std"] + normalizer.params["rho"]["mean"]
    T_val = x_np[:, 1] * normalizer.params["T"]["std"] + normalizer.params["T"]["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, k, name in zip(axes, [0, 1], ["P", "e"]):
        sc = ax.scatter(log_rho, T_val, c=err[:, k], s=2, cmap="viridis")
        plt.colorbar(sc, ax=ax, label="Абсолютная ошибка (норм.)")
        ax.set_xlabel("log10(rho)")
        ax.set_ylabel("T")
        ax.set_title(f"Ошибка предсказания {name}")
    plt.tight_layout()
    plt.savefig("error_heatmap_vanilla.png", dpi=150)
    plt.show()

def plot_violation_map(model, X_test, normalizer):
    model.eval()
    x = X_test.clone().requires_grad_(True)
    y = model(x)
    dP = torch.autograd.grad(y[:, 0].sum(), x, retain_graph=True)[0]
    dE = torch.autograd.grad(y[:, 1].sum(), x)[0]

    y_np = y.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()

    P_phys = np.power(10.0, y_np[:, 0] * normalizer.params["P"]["std"] + normalizer.params["P"]["mean"]) - 1e-12
    e_phys = y_np[:, 1] * normalizer.params["E"]["std"] + normalizer.params["E"]["mean"]

    bad_P = P_phys <= 0
    bad_dP = dP[:, 0].detach().cpu().numpy() < 0
    bad_dE = dE[:, 1].detach().cpu().numpy() < 0
    bad = bad_P | bad_dP | bad_dE

    print("\n Нарушения физики на тесте ")
    print(f"P <= 0:           {bad_P.sum()}")
    print(f"dP/drho < 0:      {bad_dP.sum()} ({100 * bad_dP.mean():.2f}%)")
    print(f"dE/dT < 0:        {bad_dE.sum()} ({100 * bad_dE.mean():.2f}%)")

    if bad.sum() == 0:
        print("Нарушений устойчивости нет.")
        return

    log_rho = x_np[:, 0] * normalizer.params["rho"]["std"] + normalizer.params["rho"]["mean"]
    T_val = x_np[:, 1] * normalizer.params["T"]["std"] + normalizer.params["T"]["mean"]

    plt.figure(figsize=(8, 6))
    plt.scatter(log_rho[~bad], T_val[~bad], s=1, c="lightgray", label="OK", rasterized=True)
    plt.scatter(log_rho[bad], T_val[bad], s=3, c="red", label=f"Нарушение ({bad.sum()})", rasterized=True)
    plt.xlabel("log10(rho)")
    plt.ylabel("T")
    plt.title("Карта нарушений устойчивости (тест)")
    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig("violation_map_vanilla.png", dpi=150)
    plt.show()


# МЕТРИКИ ДЛЯ ТАБЛИЦЫ
def evaluate_vanilla(model, X_test, Y_test, normalizer, n_speed=100_000,
                     out_path="vanilla_metrics.json"):
    p = normalizer.params
    model.eval()

    with torch.no_grad():
        pred = model(X_test).cpu().numpy()
    true = Y_test.cpu().numpy()

    # физические единицы
    P_pred = np.power(10.0, pred[:, 0] * p["P"]["std"] + p["P"]["mean"]) - 1e-12
    P_true = np.power(10.0, true[:, 0] * p["P"]["std"] + p["P"]["mean"]) - 1e-12
    E_pred = pred[:, 1] * p["E"]["std"] + p["E"]["mean"]
    E_true = true[:, 1] * p["E"]["std"] + p["E"]["mean"]

    # MAPE
    mape_P = np.mean(np.abs((P_pred - P_true) / np.abs(P_true))) * 100
    mape_E = np.mean(np.abs((E_pred - E_true) / np.abs(E_true))) * 100

    # R2 по P в лог-шкале (P меняется на 12 порядков)
    ss_res = np.sum((true[:, 0] - pred[:, 0]) ** 2)
    ss_tot = np.sum((true[:, 0] - true[:, 0].mean()) ** 2)
    r2_P = 1 - ss_res / ss_tot

    # скорость
    x = X_test[:n_speed]
    with torch.no_grad():
        model(x)  # прогрев
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    speed = len(x) / dt

    # размер
    size_mb = os.path.getsize(MODEL_FILE) / 1024 ** 2

    res = {
        "MAPE_P_percent": float(mape_P),
        "MAPE_E_percent": float(mape_E),
        "R2_P": float(r2_P),
        "size_mb": float(size_mb),
        "speed_queries_per_sec": float(speed),
    }

    print("\n Метрики Vanilla (для итоговой таблицы) ")
    print(f"MAPE P (%):        {mape_P:.4f}")
    print(f"MAPE E (%):        {mape_E:.4f}")
    print(f"R2 (P):            {r2_P:.4f}")
    print(f"Размер (МБ):       {size_mb:.4f}")
    print(f"Скорость (зап/с):  {speed:,.0f}")

    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    return res


def save_to_torchscript(model, device, out_path="model_scripted.pt"):
    """
    Сохраняет модель в TorchScript.
    """
    print(f"\nЭкспорт в TorchScript: {out_path}")
    model.eval()
    dummy_input = torch.randn(1, 2, device=device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(out_path)

    size_mb = os.path.getsize(out_path) / 1024 ** 2
    print(f"Размер: {size_mb:.4f} МБ")
    return traced_model


# ЗАПУСК
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("Устройство:", DEVICE)

    print("Загрузка данных")
    ds = load_flat_dataset()

    print("Разбиение и нормализация")
    prepared = split_and_prepare(ds)

    X_train = torch.stack([torch.tensor(prepared["train"]["rho"], dtype=torch.float32),
                           torch.tensor(prepared["train"]["T"], dtype=torch.float32)], dim=1).to(DEVICE)
    Y_train = torch.stack([torch.tensor(prepared["train"]["P"], dtype=torch.float32),
                           torch.tensor(prepared["train"]["E"], dtype=torch.float32)], dim=1).to(DEVICE)

    X_test = torch.stack([torch.tensor(prepared["test"]["rho"], dtype=torch.float32),
                          torch.tensor(prepared["test"]["T"], dtype=torch.float32)], dim=1).to(DEVICE)
    Y_test = torch.stack([torch.tensor(prepared["test"]["P"], dtype=torch.float32),
                          torch.tensor(prepared["test"]["E"], dtype=torch.float32)], dim=1).to(DEVICE)

    normalizer = prepared["normalizer"]

    model = MLP(in_dim=2, out_dim=2, width=64, depth=4).to(DEVICE)


    train_losses, test_losses = train_model(model, X_train, Y_train, X_test, Y_test)

    print("Производные")
    pts, grads = compute_derivatives(model, X_test, n_points=5)
    print_derivatives(pts, grads)


    plot_learning_curve(train_losses, test_losses)
    plot_error_heatmap(model, X_test, Y_test, normalizer)
    plot_violation_map(model, X_test, normalizer)


    evaluate_vanilla(model, X_test, Y_test, normalizer)

    evaluate_vanilla(model, X_test, Y_test, normalizer)
    save_to_torchscript(model, DEVICE, "vanilla_scripted.pt")
