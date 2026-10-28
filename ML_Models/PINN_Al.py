import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from MLP import MLP, grad
from src.data_loader import load_flat_dataset
from src.preprocessor import split_and_prepare
from PINN_Model_Trainer import PINN_Model_Trainer

import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print("Device:", device)


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




print("Загрузка данных...")
ds = load_flat_dataset()

print("Разбиение и нормализация...")
prepared = split_and_prepare(ds)

train_data = prepared["train"]
test_data = prepared["test"]
normalizer = prepared["normalizer"]

print(f"Train size: {len(train_data['rho'])}")
print(f"Test size: {len(test_data['rho'])}")

# Инициализация модели
model = MLP(in_dim=2, out_dim=2).to(device)

# Тренер
trainer = PINN_Model_Trainer(model=model, train_data=train_data, test_data=test_data,
                             normalizer=normalizer, n_iters=10_000, batch_size=4096)

# Обучение и визуализация
history = trainer.train_model(desc="PINN EOS")
trainer.visualization_train_res()