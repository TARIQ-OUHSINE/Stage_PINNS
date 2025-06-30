import torch
from torch.autograd import grad, Function


def step_function(threshold, value_before, value_after):
    class StepFunction(Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            with torch.no_grad():
                step01 = torch.heaviside(input - threshold, values=torch.zeros(1))
                return step01 * (value_after - value_before) + value_before

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output * 0

    return StepFunction


def smooth_step_function(threshold, value_before, value_after, steepness=100.0):
    def func(r):
        s = torch.sigmoid(steepness * (r - threshold))
        return (1 - s) * value_before + s * value_after
    return func

