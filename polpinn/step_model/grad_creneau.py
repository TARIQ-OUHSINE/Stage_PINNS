import torch
from torch.nn import Module
from torch.autograd import grad

import matplotlib.pyplot as plt


class RampeSeuil(Module):
    def __init__(self, threshold):
        Module.__init__(self)
        self.threshold = threshold

    def __call__(self, x):
        return (x > self.threshold) * x


class Creneau(Module):
    def __init__(self, threshold):
        Module.__init__(self)
        self.threshold = threshold

    def __call__(self, x):
        return (x > self.threshold) * (x**2 + 1) / (x**2 + 1)


class CreneauBug(Module):
    def __init__(self, threshold):
        Module.__init__(self)
        self.threshold = threshold

    def __call__(self, x):
        return (x > self.threshold) * torch.ones_like(x, requires_grad=True)


class CreneauHeaviside(Module):
    def __init__(self, threshold):
        Module.__init__(self)
        self.threshold = threshold

    def __call__(self, x):
        with torch.no_grad():
            return torch.heaviside(x - self.threshold, values=torch.zeros(1))


for f in (
    RampeSeuil(1 / 3),
    Creneau(1 / 3),
    CreneauBug(1 / 3),
    CreneauHeaviside(1 / 3),
):
    try:
        x_tensor = torch.linspace(0, 1, 20, requires_grad=True)
        y_list = []
        dydx_list = []
        for x in x_tensor:
            y = f(x)
            dydx = grad(outputs=y, inputs=x)
            print(x, y, dydx)
            y_list.append(y.detach().numpy())
            dydx_list.append(dydx)

        x_array = x_tensor.detach().numpy()
        print(x_array)
        fig, axes = plt.subplots(2, 1)
        axes[0].scatter(x_array, y_list)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[1].scatter(x_array, dydx_list)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("dy/dx")
    except RuntimeError as e:
        print(f, e)
plt.show()
