from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


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

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                if "first_moment" not in state:
                    state["first_moment"] = torch.zeros(grad.shape)
                if "second_moment" not in state:
                    state["second_moment"] = torch.zeros(grad.shape)
                if "timestep" not in state:
                    state["timestep"] = 0

                beta1, beta2 = self.defaults["betas"][0], self.defaults["betas"][1]
                state["timestep"] += 1
                state["first_moment"] = beta1 * state["first_moment"] + (1 - beta1) * grad
                state["second_moment"] = beta2 * state["second_moment"] + (1 - beta2) * grad * grad

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if self.defaults["correct_bias"]:
                    m_hat = state["first_moment"] / (1 - beta1 ** state["timestep"])
                    v_hat = state["second_moment"] / (1 - beta2 ** state["timestep"])
                else:
                    m_hat, v_hat = state["first_moment"], state["second_moment"]

                # Update parameters
                eps = self.defaults["eps"]
                lam = self.defaults["weight_decay"]
                p.data -= alpha * m_hat / (torch.sqrt(v_hat) + eps) + lam * p.data

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss