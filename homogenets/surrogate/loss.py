import torch


class MSELoss(torch.nn.Module):
    def __init__(self, batch_size: int, feature_size: int):
        super(MSELoss, self).__init__()
        self.__Y: torch.Tensor = torch.empty((batch_size, feature_size))
        self.__Yhat: torch.Tensor = torch.empty((batch_size, feature_size))

    @torch.jit.export
    def evaluate(self, Yhat, Y):
        self.__Y = Y  # target value Y
        self.__Yhat = Yhat
        J = (Y - Yhat) ** 2
        if J.dim() > 1:
            J = torch.mean(J, 1)  # handles multiple target values as a mean
        return torch.mean(J, 0)  # returns the batch average

    def gradient(self):
        N = self.__Y.shape[0]  # batch size
        if self.__Y.dim() > 1:
            F = self.__Y.shape[1]  # output features
        else:
            F = 1
        dJ = -2 * (self.__Y - self.__Yhat) / N / F
        return dJ


class RL2Loss(torch.nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def forward(self, yhat, y):
        # num = torch.abs(yhat * y)
        # den = torch.norm(yhat) * torch.norm(y)
        # return torch.sum(num/den)
        # return torch.norm(yhat - y)/torch.norm(y)
        return torch.sum(
            torch.linalg.vector_norm(y - yhat, dim=1, ord=self.order)
            / torch.linalg.vector_norm(y, dim=1, ord=self.order)
        )


class RelLoss(torch.nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def forward(self, yhat, y):
        rel_error = torch.abs((y - yhat) / y)
        return torch.sum(torch.linalg.vector_norm(rel_error, dim=1, ord=self.order))


class AngleLoss(torch.nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def forward(self, yhat, y):
        rl2 = torch.sum(
            torch.linalg.vector_norm(y - yhat, dim=1, ord=self.order)
            / torch.linalg.vector_norm(y, dim=1, ord=self.order)
        )

        # dot = torch.sum(y*yhat, dim=1)
        # norms = torch.linalg.vector_norm(y, dim=1) * torch.linalg.vector_norm(yhat, dim=1)
        # angle = torch.sum(dot/norms)
        angle = torch.sum((self.cos(y, yhat)) ** 2)
        return rl2 + angle


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        # num = torch.abs(yhat * y)
        # den = torch.norm(yhat) * torch.norm(y)
        # return torch.sum(num/den)
        # return torch.norm(yhat - y)/torch.norm(y)
        num = torch.sum((y - yhat) ** 2, dim=1)
        den = torch.sum(y**2, dim=1)
        return torch.sum(num / den)
