import torch
import torch.nn as nn

from ML_Models.PINN_Al import device


class MLP(nn.Module):
    """Полносвязная нейронная сеть (MLP) с настраиваемой архитектурой.

        Сеть состоит из `depth` скрытых слоёв с заданной шириной `width` и
        выбранной функцией активации, за которыми следует линейный выходной слой.
        Веса инициализируются методом Ксавье, смещения — нулями.
    """
    def __init__(self, in_dim=2, out_dim=2, width=128, depth=5, activation=nn.GELU):
        """Инициализирует MLP.

        Args:
            in_dim (int, optional): Размерность входного признака. По умолчанию 2.
            out_dim (int, optional): Размерность выходного признака. По умолчанию 2.
            width (int, optional): Ширина скрытых слоёв. По умолчанию 128.
            depth (int, optional): Количество скрытых слоёв. По умолчанию 5.
            activation (callable, optional): Класс функции активации (например, nn.GELU).
                По умолчанию nn.GELU.
        """
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), activation()]
            d = width
        layers.append(nn.Linear(d, out_dim))          # linear output layer
        self.net = nn.Sequential(*layers)
        for m in self.net:                            # Xavier initialisation, zero biases
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt):
        """Выполняет прямой проход.

        Args:
            xt (torch.Tensor): Входной тензор размера [batch_size, in_dim].

        Returns:
            torch.Tensor: Выходной тензор размера [batch_size, out_dim].
        """
        return self.net(xt)


def grad(outputs, inputs):
    """Вычисляет градиент выходов по входам с сохранением графа вычислений.

    Args:
        outputs (torch.Tensor): Тензор выходов, для которого вычисляется градиент.
        inputs (torch.Tensor): Тензор входов, по которому вычисляется градиент.

    Returns:
        torch.Tensor: Градиент `outputs` по `inputs` той же формы, что и `inputs`.
    """
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True)[0]
