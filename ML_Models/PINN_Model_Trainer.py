import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from Loss import eos_loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN_Model_Trainer:
    def __init__(self, model, train_data, test_data, normalizer, n_iters=5000, batch_size=2048):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.normalizer = normalizer
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.n_train = len(train_data['rho'])

        # Предварительно создаем тензоры для быстрой выборки батчей
        self.rho_t = torch.tensor(train_data['rho'], dtype=torch.float32).unsqueeze(1).to(device)
        self.T_t = torch.tensor(train_data['T'], dtype=torch.float32).unsqueeze(1).to(device)
        self.p_t = torch.tensor(train_data['P'], dtype=torch.float32).unsqueeze(1).to(device)
        self.e_t = torch.tensor(train_data['E'], dtype=torch.float32).unsqueeze(1).to(device)

    def train(self, loss_fn, n_iters, lr=1e-3, extra_params=(), eval_fn=None, log_every=500, desc="train",
              scale={'pde': 1.0, 'bc': 1.0, 'ic': 0.01, 'L_C': 0.01, 'L_th': 0.01}):
        params = list(self.model.parameters()) + list(extra_params)
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9 ** (1 / 1000))
        history = {"it": []}

        for it in tqdm(range(1, n_iters + 1), desc=desc, leave=False):
            terms = loss_fn()
            for key in terms:
                terms[key] = terms[key] * scale[key]

            loss = sum(terms.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if it == 1 or it % log_every == 0:
                history["it"].append(it)
                for k, v in terms.items():
                    history.setdefault("loss_" + k, []).append(v.item())
                if eval_fn is not None:
                    for k, v in eval_fn(self.model).items():
                        history.setdefault(k, []).append(v)
        return history

    def rel_l2(self, pred, exact):
        return float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-12))

    @torch.no_grad()
    def predict_eos(self, model, rho_np, T_np):
        rho_t = torch.tensor(rho_np, dtype=torch.float32, device=device).unsqueeze(1)
        T_t = torch.tensor(T_np, dtype=torch.float32, device=device).unsqueeze(1)
        preds = model(rho_t, T_t).cpu().numpy()
        return preds[:, 0], preds[:, 1]

    def eval_fn(self, m):
        p_pred, e_pred = self.predict_eos(m, self.test_data['rho'], self.test_data['T'])
        P_pred_phys = self.normalizer.inverse_transform_P(p_pred)
        E_pred_phys = self.normalizer.inverse_transform_E(e_pred)

        P_test_phys = self.normalizer.inverse_transform_P(self.test_data['P'])
        E_test_phys = self.normalizer.inverse_transform_E(self.test_data['E'])

        return {
            "error_P": self.rel_l2(P_pred_phys, P_test_phys),
            "error_E": self.rel_l2(E_pred_phys, E_test_phys)
        }

    def train_model(self, desc="EOS PINN"):
        torch.manual_seed(0)

        def loss_fn():
            idx = torch.randint(0, self.n_train, (self.batch_size,), device=device)
            return eos_loss_fn(
                self.model,
                self.rho_t[idx], self.T_t[idx], self.p_t[idx], self.e_t[idx],
                self.normalizer
            )

        start = time.time()
        history = self.train(loss_fn, self.n_iters, eval_fn=self.eval_fn, desc=desc)
        print(f"Training took {time.time() - start:.0f} s")
        print(f"Relative L2 error P: {history['error_P'][-1]:.2e}")
        print(f"Relative L2 error E: {history['error_E'][-1]:.2e}")

        # Визуализация лоссов
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for k, v in history.items():
            if k != "it":
                ax.semilogy(history["it"], np.abs(v), label=k)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("EOS PINN Training History")
        plt.tight_layout()
        plt.show()
        return history

    def visualization_train_res(self):
        p_pred, e_pred = self.predict_eos(self.model, self.test_data['rho'], self.test_data['T'])
        P_pred_phys = self.normalizer.inverse_transform_P(p_pred)
        E_pred_phys = self.normalizer.inverse_transform_E(e_pred)

        P_test_phys = self.normalizer.inverse_transform_P(self.test_data['P'])
        E_test_phys = self.normalizer.inverse_transform_E(self.test_data['E'])

        # Восстановление физических rho и T для осей
        rho_phys = 10 ** (self.test_data['rho'] * self.normalizer.params["rho"]["std"] + self.normalizer.params["rho"][
            "mean"]) - 1e-12
        T_phys = self.test_data['T'] * self.normalizer.params["T"]["std"] + self.normalizer.params["T"]["mean"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Давление
        sc = axes[0].scatter(rho_phys, T_phys, c=P_test_phys, cmap='viridis', alpha=0.7, s=5)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_title("P exact (GPa)")
        axes[0].set_xlabel("rho")
        axes[0].set_ylabel("T")
        plt.colorbar(sc, ax=axes[0])

        # Ошибка давления
        err_P = np.abs(P_pred_phys - P_test_phys) / (np.abs(P_test_phys) + 1e-8)
        sc = axes[1].scatter(rho_phys, T_phys, c=err_P, cmap='Reds', alpha=0.7, s=5)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title("P relative error")
        axes[1].set_xlabel("rho")
        plt.colorbar(sc, ax=axes[1])

        # Ошибка энергии
        err_E = np.abs(E_pred_phys - E_test_phys) / (np.abs(E_test_phys) + 1e-8)
        sc = axes[2].scatter(rho_phys, T_phys, c=err_E, cmap='Blues', alpha=0.7, s=5)
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].set_title("E relative error")
        axes[2].set_xlabel("rho")
        plt.colorbar(sc, ax=axes[2])

        plt.tight_layout()
        plt.show()

