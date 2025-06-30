from torch import nn, tanh, sigmoid
import torch


class G_MLP(nn.Module):
    def __init__(
        self,
        nb_layer: int,
        hidden_layer: int,
    ):
        super(G_MLP, self).__init__()
        self.fc_int = nn.Linear(2, hidden_layer)
        self.fc = nn.ModuleList(
            [nn.Linear(hidden_layer, hidden_layer) for _ in range(nb_layer)]
        )
        self.fc_out = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        x = tanh(self.fc_int(x))
        for fc in self.fc:
            x = tanh(fc(x))
        x = sigmoid(self.fc_out(x))
        return x


class PointPolarization:
    def __init__(self, G):
        self.G = G

    def __call__(self, X):
        X.requires_grad_(True)
        r = X[:, 0].view(-1, 1)
        g_tensor = self.G(X)
        dG_d = torch.autograd.grad(
            g_tensor, X, grad_outputs=torch.ones_like(g_tensor), create_graph=True
        )[0]
        dG_dr = dG_d[:, 0].view(-1, 1)
        P_tensor = (r / 3) * dG_dr + g_tensor
        return P_tensor
