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
    """Обёртка над MLP для учёта нормализации входных и выходных переменных.

    Класс принимает нормализованные плотность и температуру, объединяет их
    и передаёт в базовую модель. Содержит параметры нормализации для
    обратного преобразования предсказанных давления и энергии.
    """
    def __init__(self, model, data_scale):
        """Инициализирует обёртку PINN_EOS.

        Args:
            model (nn.Module): Базовая нейронная сеть, принимающая тензор
                размера [batch_size, 2] и возвращающая [batch_size, 2].
            data_scale (dict): Словарь с параметрами нормализации:
                - 'rho_mean' (float): Среднее значение плотности.
                - 'rho_std' (float): Стандартное отклонение плотности.
                - 'T_mean' (float): Среднее значение температуры.
                - 'T_std' (float): Стандартное отклонение температуры.
                - 'P_scale' (float): Масштаб для давления.
                - 'E_scale' (float): Масштаб для энергии.
        """
        super().__init__()
        self.model = model
        self.rho_mean, self.rho_std = data_scale['rho_mean'], data_scale['rho_std']
        self.T_mean, self.T_std = data_scale['T_mean'], data_scale['T_std']
        self.P_scale = data_scale['P_scale']
        self.E_scale = data_scale['E_scale']

    def forward(self, rho_n, T_n):
        """Выполняет прямой проход через базовую модель.

        Args:
            rho_n (torch.Tensor): Нормализованная плотность, форма [batch_size, 1].
            T_n (torch.Tensor): Нормализованная температура, форма [batch_size, 1].
        
        Returns:
            torch.Tensor: Предсказанные нормализованные давление и энергия,
                форма [batch_size, 2].
        """
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
