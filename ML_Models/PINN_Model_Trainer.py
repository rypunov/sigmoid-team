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


class PINN_Model_Trainer:
    def __init__(self, model, train_loader, val_loader, exact_data):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Для визуализации (сохраняем эталонные данные)
        self.rho_log_np, self.T_log_np, self.rho_np, self.T_np, self.p_np, self.e_np = exact_data

    def train(self, n_iters, lr=1e-3, eval_fn=None, log_every=500, desc="train",
              scale={'pde': 1.0, 'bc': 1.0, 'ic': 1.0, 'L_C': 0.01, 'L_th': 0.01}):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9 ** (1 / 1000))
        history = {"it": []}

        for it in tqdm(range(1, n_iters + 1), desc=desc, leave=False):
            for batch in self.train_loader:
                rho_log, T_log, rho, T, p, e = [x.to(device) for x in batch]

                optimizer.zero_grad()
                terms = eos_loss_fn(self.model, rho_log, T_log, rho, T, p, e)

                loss = terms["total"]
                loss.backward()
                optimizer.step()

            scheduler.step()

            if it == 1 or it % log_every == 0:
                history["it"].append(it)
                with torch.no_grad():
                    for k, v in terms.items():
                        if k != "total":
                            history.setdefault("loss_" + k, []).append(v.item())
                    history.setdefault("loss_total", []).append(loss.item())

                if eval_fn is not None:
                    for k, v in eval_fn(self.model).items():
                        history.setdefault(k, []).append(v)
        return history

    def rel_l2(self, pred, exact):
        return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))

    @torch.no_grad()
    def predict_grid(self, model, rho_log, T_log):
        rho_log_t = torch.tensor(rho_log, dtype=torch.float32, device=device).unsqueeze(1)
        T_log_t = torch.tensor(T_log, dtype=torch.float32, device=device).unsqueeze(1)
        preds = model(rho_log_t, T_log_t).cpu().numpy()
        return preds[:, 0], preds[:, 1]  # p_pred, e_pred

    def plot_history(self, history, title=""):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for k, v in history.items():
            if k != "it":
                ax.semilogy(history["it"], np.abs(v), label=k)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def train_model(self, desc="EOS PINN"):
        torch.manual_seed(0)

        # Простая валидация для отслеживания ошибки на тестовых точках
        eval_fn = lambda m: {"val_rel_err_p": self._calc_val_error(m)[0],
                             "val_rel_err_e": self._calc_val_error(m)[1]}

        start = time.time()
        history = self.train(n_iters=5000, eval_fn=eval_fn, desc=desc)
        print(f"Training took {time.time() - start:.0f} s")
        self.plot_history(history, "EOS: loss terms and validation error")
        return history

    def _calc_val_error(self, model):
        # Считаем ошибку по всем данным для простоты (в идеале по val_loader)
        p_pred, e_pred = self.predict_grid(model, self.rho_log_np, self.T_log_np)
        err_p = self.rel_l2(p_pred.flatten(), self.p_np)
        err_e = self.rel_l2(e_pred.flatten(), self.e_np)
        return err_p, err_e

    def visualization_train_res(self):
        p_pred, e_pred = self.predict_grid(self.model, self.rho_log_np, self.T_log_np)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Давление
        ax = axes[0]
        sc = ax.scatter(self.rho_np, self.T_np, c=self.p_np, cmap='viridis', alpha=0.5, s=10)
        ax.set_xscale('log');
        ax.set_yscale('log')
        ax.set_title("Точное давление P");
        ax.set_xlabel("rho");
        ax.set_ylabel("T")
        plt.colorbar(sc, ax=ax)

        # Ошибка давления
        ax = axes[1]
        err_p = np.abs(p_pred.flatten() - self.p_np) / (np.abs(self.p_np) + 1e-8)
        sc = ax.scatter(self.rho_np, self.T_np, c=err_p, cmap='Reds', alpha=0.5, s=10)
        ax.set_xscale('log');
        ax.set_yscale('log')
        ax.set_title("Отн. ошибка P");
        ax.set_xlabel("rho")
        plt.colorbar(sc, ax=ax)

        # Ошибка энергии
        ax = axes[2]
        err_e = np.abs(e_pred.flatten() - self.e_np) / (np.abs(self.e_np) + 1e-8)
        sc = ax.scatter(self.rho_np, self.T_np, c=err_e, cmap='Blues', alpha=0.5, s=10)
        ax.set_xscale('log');
        ax.set_yscale('log')
        ax.set_title("Отн. ошибка e");
        ax.set_xlabel("rho")
        plt.colorbar(sc, ax=ax)

        plt.tight_layout()
        plt.show()