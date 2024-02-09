from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros(grad.shape)
                    state["second_moment"] = torch.zeros(grad.shape)

                state["step"] += 1
                beta1, beta2 = group["betas"]
                first_moment, second_moment = state["first_moment"], state["second_moment"]

                # weight decay
                first_moment = first_moment * beta1 + (1.0 - beta1) * grad
                second_moment = second_moment * beta2 + (1.0 - beta2) * grad * grad
                state["first_moment"], state["second_moment"] = first_moment, second_moment
                denom = second_moment.sqrt() + self.defaults["eps"]

                # correct for bias
                adj_alpha = alpha
                if self.defaults["correct_bias"]:
                    beta1_corr = 1.0 - beta1 ** state["step"]
                    beta2_corr = 1.0 - beta2 ** state["step"]
                    adj_alpha = alpha * math.sqrt(beta2_corr) / beta1_corr

                p.data -= adj_alpha * first_moment / denom
                p.data -= alpha * group["weight_decay"] * p.data

        return loss