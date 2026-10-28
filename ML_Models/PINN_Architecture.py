class PINN_Architecture():
    def __init__(self, model, N_F=10_000, N_B=500, N_0=1_000):
        self.N_F = N_F
        self.N_B = N_B
        self.N_0 = N_0
        self.model = model

    def residual(self, x, t):
        pass

    def loss(self):
        pass

