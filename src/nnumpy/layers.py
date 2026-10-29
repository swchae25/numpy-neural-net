"""Layers with explicit forward and backward passes.

Convention: inputs are (batch, features). ``backward(grad_out)`` receives dL/d(output),
stores parameter gradients in ``self.grads`` and returns dL/d(input).
"""
from __future__ import annotations

import numpy as np


class Layer:
    params: dict[str, np.ndarray]
    grads: dict[str, np.ndarray]

    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer y = x W + b with He or Xavier initialisation."""

    def __init__(self, n_in: int, n_out: int, init: str = "he", rng: np.random.Generator | None = None):
        super().__init__()
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / n_in) if init == "he" else np.sqrt(1.0 / n_in)
        self.params = {"W": rng.normal(0.0, scale, (n_in, n_out)), "b": np.zeros(n_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x, training=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out):
        self.grads["W"] = self.x.T @ grad_out
        self.grads["b"] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T


class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class Tanh(Layer):
    def forward(self, x, training=True):
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad_out):
        return grad_out * (1.0 - self.y**2)


class Sigmoid(Layer):
    def forward(self, x, training=True):
        # Numerically stable for large |x|.
        self.y = np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))
        return self.y

    def backward(self, grad_out):
        return grad_out * self.y * (1.0 - self.y)
