import torch
import torch.nn as nn

class MLP(nn.Module):
    """Полносвязная нейронная сеть.

    Args:
        in_dim (int): Размерность входного вектора.
        out_dim (int, optional): Размерность выхода. Default: 2.
        width (int, optional): Число нейронов в скрытых слоях. Default: 128.
        depth (int, optional): Количество скрытых слоёв. Default: 5.
        activation (type[nn.Module], optional): Класс функции активации,
            например ``nn.GELU`` или ``nn.ReLU``. Default: ``nn.GELU``.
    """
    def __init__(self, in_dim, out_dim=2, width=128, depth=5, activation=nn.GELU):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), activation()]
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt):
        """Прямой проход сети.
        Args:
            xt (torch.Tensor): Входной тензор формы (batch, in_dim).
        Returns:
            torch.Tensor: Выходной тензор формы (batch, out_dim).
        """
        return self.net(xt)


def grad(outputs, inputs):
    """Вычисляет градиент суммы выходов по входам с сохранением графа.
        Args:
            outputs (torch.Tensor): Тензор выходов,
                для которого считается градиент.
            inputs (torch.Tensor): Тензор входов, по которому берётся производная.
        Returns:
            torch.Tensor: Градиент той же формы, что и inputs.
        """
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True)[0]
