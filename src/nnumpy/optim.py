"""Optimisers update parameters in place from the gradients stored in each layer."""
from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, lr: float = 0.1, momentum: float = 0.0, weight_decay: float = 0.0):
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.velocity: dict[int, np.ndarray] = {}

    def step(self, layers):
        for layer in layers:
            for k, p in layer.params.items():
                g = layer.grads[k] + (self.weight_decay * p if k == "W" else 0.0)
                v = self.velocity.setdefault(id(p), np.zeros_like(p))
                v *= self.momentum
                v -= self.lr * g
                p += v


class Adam:
    """Adam with decoupled weight decay (AdamW) applied to weight matrices only."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.lr, self.b1, self.b2, self.eps, self.weight_decay = lr, beta1, beta2, eps, weight_decay
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            for k, p in layer.params.items():
                g = layer.grads[k]
                m = self.m.setdefault(id(p), np.zeros_like(p))
                v = self.v.setdefault(id(p), np.zeros_like(p))
                m[:] = self.b1 * m + (1 - self.b1) * g
                v[:] = self.b2 * v + (1 - self.b2) * g * g
                m_hat = m / (1 - self.b1**self.t)
                v_hat = v / (1 - self.b2**self.t)
                if self.weight_decay and k == "W":
                    p -= self.lr * self.weight_decay * p
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
