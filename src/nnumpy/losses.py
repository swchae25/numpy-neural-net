"""Losses return the mean loss over the batch; ``backward()`` returns dL/d(logits or outputs)."""
from __future__ import annotations

import numpy as np


class SoftmaxCrossEntropy:
    """Softmax + cross-entropy fused, so the gradient is simply (p - y) / n."""

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        z = logits - logits.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        self.p = np.exp(log_p)
        self.labels = labels
        return float(-log_p[np.arange(len(labels)), labels].mean())

    def backward(self) -> np.ndarray:
        g = self.p.copy()
        g[np.arange(len(self.labels)), self.labels] -= 1.0
        return g / len(self.labels)


class BinaryCrossEntropy:
    """BCE on raw logits (sigmoid folded in) for numerical stability."""

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.logits, self.t = logits, targets.reshape(logits.shape)
        # log(1 + exp(-|z|)) + max(z, 0) - z * t
        loss = np.logaddexp(0.0, logits) - logits * self.t
        return float(loss.mean())

    def backward(self) -> np.ndarray:
        p = 0.5 * (1.0 + np.tanh(0.5 * self.logits))  # stable sigmoid
        return (p - self.t) / self.logits.size


class MeanSquaredError:
    def forward(self, y: np.ndarray, target: np.ndarray) -> float:
        self.diff = y - target.reshape(y.shape)
        return float((self.diff**2).mean())

    def backward(self) -> np.ndarray:
        return 2.0 * self.diff / self.diff.size
