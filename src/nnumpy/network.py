"""A sequential network with mini-batch training and early stopping."""
from __future__ import annotations

import copy

import numpy as np


class Network:
    def __init__(self, layers, loss):
        self.layers, self.loss = list(layers), loss
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def loss_and_grads(self, x, y) -> float:
        value = self.loss.forward(self.forward(x, training=True), y)
        self.backward(self.loss.backward())
        return value

    def predict(self, x):
        out = self.forward(x)
        return out.argmax(axis=1) if out.shape[1] > 1 else (out[:, 0] > 0).astype(int)

    def accuracy(self, x, y) -> float:
        return float((self.predict(x) == y.ravel()).mean())

    def fit(self, x, y, optimizer, epochs=100, batch_size=32, x_val=None, y_val=None,
            patience: int | None = None, rng: np.random.Generator | None = None, verbose=False):
        """Mini-batch training. With ``patience`` and validation data, stops early and
        restores the parameters from the epoch with the lowest validation loss."""
        rng = rng or np.random.default_rng(0)
        best, best_params, waited = np.inf, None, 0
        for epoch in range(epochs):
            order = rng.permutation(len(x))
            losses = []
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                losses.append(self.loss_and_grads(x[idx], y[idx]) * len(idx))
                optimizer.step(self.layers)
            self.history["train_loss"].append(sum(losses) / len(x))
            if x_val is not None:
                val = self.loss.forward(self.forward(x_val), y_val)
                self.history["val_loss"].append(val)
                self.history["val_acc"].append(self.accuracy(x_val, y_val))
                if val < best - 1e-6:
                    best, waited = val, 0
                    best_params = copy.deepcopy([l.params for l in self.layers])
                else:
                    waited += 1
                if patience is not None and waited >= patience:
                    break
            if verbose and epoch % 10 == 0:
                print(f"epoch {epoch:4d}  train {self.history['train_loss'][-1]:.4f}")
        if best_params is not None and patience is not None:
            for layer, params in zip(self.layers, best_params):
                for k in layer.params:
                    layer.params[k][...] = params[k]
        return self.history
