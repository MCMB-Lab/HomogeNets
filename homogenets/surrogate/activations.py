import torch


class Activation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def update(self, eta):
        pass


class IdentityActivation(Activation):
    def __init__(self, batch_size: int, output_size: int):
        super(IdentityActivation, self).__init__()
        self.weight = torch.ones((batch_size, output_size))

    def forward(self, X):
        self.weight = torch.diag_embed(torch.ones(X.shape))
        return X

    def backward(self, G):
        return G


class SigmoidActivation(Activation):
    def __init__(self, batch_size: int, output_size: int):
        super().__init__()
        # self.Y = torch.jit.Attribute(torch.empty((batch_size, output_size)), torch.Tensor)
        self.Y = torch.empty((batch_size, output_size))
        self.weight = torch.empty((batch_size, output_size))

    def forward(self, X):
        Y = torch.sigmoid(X)
        self.Y = Y
        self.weight = torch.diag_embed(self.gradient())
        return Y

    def gradient(self):
        dY = self.Y * (1 - self.Y)
        return dY

    # @torch.jit.script
    def backward(self, G):
        return self.gradient() * G


class TanhActivation(Activation):
    def __init__(self, batch_size: int, output_size: int):
        super(TanhActivation, self).__init__()
        self.Y = torch.empty((batch_size, output_size))
        self.weight = torch.empty((batch_size, output_size))

    def forward(self, X):
        Y = torch.tanh(X)
        self.Y = Y
        self.weight = torch.diag_embed(self.gradient())
        return Y

    def gradient(self):
        dY = 1 - ((self.Y) ** 2)
        return dY

    def backward(self, G):
        return self.gradient() * G


class ZscoreActivation(Activation):
    def __init__(
        self, batch_size: int, output_size: int, means: torch.Tensor, stds: torch.Tensor
    ):
        super(ZscoreActivation, self).__init__()
        self.mu: torch.Tensor = means
        self.sigma: torch.Tensor = stds

    def forward(self, X):
        return (X - self.mu) / self.sigma

    def backward(self, G):
        return G / self.sigma

    def gradient(self):
        return 1 / self.sigma


class DeZscoreActivation(Activation):
    def __init__(
        self, batch_size: int, output_size: int, means: torch.Tensor, stds: torch.Tensor
    ):
        super(DeZscoreActivation, self).__init__()
        self.mu: torch.Tensor = means
        self.sigma: torch.Tensor = stds

    def forward(self, X):
        return X * self.sigma + self.mu

    def gradient(self):
        return self.sigma

    def backward(self, G):
        return G * self.sigma


class MaxActivation(Activation):
    def __init__(
        self, batch_size: int, output_size: int, maxs: torch.Tensor, mins: torch.Tensor
    ):
        super(MaxActivation, self).__init__()
        self.maxs: torch.Tensor = maxs
        self.mins: torch.Tensor = mins

    def forward(self, X):
        return (X - self.mins) / (self.maxs - self.mins)

    def backward(self, G):
        return G / (self.maxs - self.mins)

    def gradient(self):
        return 1.0 / (self.maxs - self.mins)


class DeMaxActivation(Activation):
    def __init__(
        self, batch_size: int, output_size: int, maxs: torch.Tensor, mins: torch.Tensor
    ):
        super(DeMaxActivation, self).__init__()
        self.maxs: torch.Tensor = maxs
        self.mins: torch.Tensor = mins

    def forward(self, X):
        return X * (self.maxs - self.mins) + self.mins

    def gradient(self):
        return self.maxs - self.mins

    def backward(self, G):
        return G * self.gradient()
