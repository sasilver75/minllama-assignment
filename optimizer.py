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

        # Looping over each parameter group
        for group in self.param_groups:
            # Pulling out hyperparameters for convenience
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            correct_bias = group["correct_bias"]

            # Looping over each parameter in the current group
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    # Boo fuck sparse gradients boo
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary -- Let's initialize it :)
                state = self.state[p]
                if len(state) == 0:
                    # Initialize our first/second moment buffers!
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Sam note: Note taht all of these trailing-_ methods mean that the operation is in-place, rather than reurning a new one.
                # Just wanted to try it out. Found a code snippet using this. Saves some memory too, but a little "dangerous"

                # Decoupled weight decay (We shrink the param towards zero)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update biased first and second moment estimates!
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # Note add x.addcmul_(y,z, value=y) computes x += v * (y*z) in place. Nifty.

                # Next let's compute bias-coprrected step size (if required)
                # Early in training, m_t and v_t are biased towards zero, and this attempts to undo that.
                if correct_bias:
                    bias_corr1 = 1.0 - beta1 ** step
                    bias_corr2 = 1.0 - beta2 ** step
                    step_size = lr * math.sqrt(bias_corr2) / bias_corr1
                else:
                    step_size = lr

                # Now we can update our params!
                # p = p - step_size(m_t/(sqrt(v_t)+eps))
                denom = exp_avg_sq.sqrt().add(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                # Similarly, note that x.addcdiv(y,z, value=v) computes x += v * (y/z) in place.

        return loss