import torch
from torch.autograd import grad, Function

import matplotlib.pyplot as plt


def rampe_seuil(threshold):
    class RampeSeuil(Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (input > threshold) * input

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output * (input > threshold)

    return RampeSeuil


def creneau(threshold):
    class Creneau(Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (input > threshold) * (input**2 + 1) / (input**2 + 1)

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output * 0

    return Creneau


def creneau_heaviside(threshold):
    class CreneauHeaviside(Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            with torch.no_grad():
                return torch.heaviside(input - threshold, values=torch.zeros(1))

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output * 0

    return CreneauHeaviside


for f in (
    rampe_seuil(1 / 3),
    creneau(1 / 3),
    creneau_heaviside(1 / 3),
):
    try:
        x_tensor = torch.linspace(0, 1, 20, requires_grad=True)
        y_list = []
        dydx_list = []
        for x in x_tensor:
            y = f.apply(x)
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
