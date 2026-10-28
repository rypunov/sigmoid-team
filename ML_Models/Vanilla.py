"""
Обучение Vanilla 
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from config import MODEL_FILE, SEED, EPOCHS, LR, BATCH_SIZE, DEVICE
from data_loader import load_flat_dataset
from preprocessor import split_and_prepare
from MLP import MLP

# ОБУЧЕНИЕ

def train_model(model, X_train, Y_train, X_test, Y_test):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
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
            total += loss.item() * len(idx)
        train_losses.append(total / n)

        model.eval()
        with torch.no_grad():
            test_losses.append(loss_fn(model(X_test), Y_test).item())

        if epoch == 1 or epoch % 50 == 0:
            print(f"Эпоха {epoch:4d}/{EPOCHS} | train MSE={train_losses[-1]:.3e} | test MSE={test_losses[-1]:.3e}")

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

#  ВИЗУАЛИЗАЦИЯ 

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
    # Восстанавливаем оси для графика (T без логарифма, rho через 10^x)
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
    
    # Обратный пересчет P (через логарифм) и E (без логарифма)
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

# ЗАПУСК

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("Устройство:", DEVICE)


    ds = load_flat_dataset()

    prepared = split_and_prepare(ds)
    
    
    try:
        X_train = prepared["train"]["X"].to(DEVICE)
        Y_train = prepared["train"]["Y"].to(DEVICE)
        X_test = prepared["test"]["X"].to(DEVICE)
        Y_test = prepared["test"]["Y"].to(DEVICE)
    except TypeError:
        # Если вернулись кортежи (X, Y)
        X_train, Y_train = prepared["train"]
        X_test, Y_test = prepared["test"]
        X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
        X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)

    normalizer = prepared["normalizer"]


    model = MLP(in_dim=2, out_dim=2, width=64, depth=4).to(DEVICE)
    

    print("обучение")
    train_losses, test_losses = train_model(model, X_train, Y_train, X_test, Y_test)


    pts, grads = compute_derivatives(model, X_test, n_points=5)
    print_derivatives(pts, grads)


    plot_learning_curve(train_losses, test_losses)
    plot_error_heatmap(model, X_test, Y_test, normalizer)
    plot_violation_map(model, X_test, normalizer)