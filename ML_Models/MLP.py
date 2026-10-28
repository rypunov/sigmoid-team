class MLP(nn.Module):
    """Полносвязная нейронная сеть"""
    def __init__(self, in_dim, out_dim=1, width=64, depth=4, activation=nn.Tanh):
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
        return self.net(xt)


def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True)[0]
