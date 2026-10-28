import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from MLP import MLP, grad
from src.data_loader import load_flat_dataset
from src.preprocessor import split_and_prepare
from PINN_Model_Trainer import PINN_Model_Trainer
from src.config import MODEL_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print("Device:", device)

print("Загрузка данных...")
ds = load_flat_dataset()

print("Разбиение и нормализация...")
prepared = split_and_prepare(ds)

train_data = prepared["train"]
test_data  = prepared["test"]
normalizer = prepared["normalizer"]

print(f"Train size: {len(train_data['rho'])}")
print(f"Test size:  {len(test_data['rho'])}")

model = MLP(in_dim=2, out_dim=2, width=128, depth=5, activation=nn.GELU).to(device)

trainer = PINN_Model_Trainer(model=model, train_data=train_data, test_data=test_data,
                             normalizer=normalizer, n_iters=30_000, batch_size=4096)

history = trainer.train_model(desc="PINN EOS")

with open(f"{MODEL_DIR}/normalizer.pkl", "wb") as f:
    pickle.dump(normalizer, f)
torch.save(model.state_dict(), f"{MODEL_DIR}/pinn_model.pt")

trainer.visualization_train_res()

fig, ax = plt.subplots(figsize=(10, 5))
for k, v in history.items():
    if k != "it" and (k.startswith("loss_") or k.startswith("error_") or k.startswith("train_error_")):
        ax.semilogy(history["it"], np.abs(v), label=k)

if history["it"]:
    ax.axvline(x=trainer.n_iters, color='red', linestyle='--', label='Switch to L-BFGS')

ax.set_xlabel("iteration")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
ax.set_title("EOS PINN Training History")
plt.show()