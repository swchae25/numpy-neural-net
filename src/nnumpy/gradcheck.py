"""Compare analytical gradients with centred finite differences."""
from __future__ import annotations

import numpy as np


def gradient_check(net, x, y, eps: float = 1e-6, samples: int = 20, rng=None) -> float:
    """Return the largest relative error over randomly sampled parameter entries.

    Relative error |a - n| / max(|a| + |n|, 1e-12); values below about 1e-5 mean
    backpropagation matches the numerical derivative.
    """
    rng = rng or np.random.default_rng(0)
    net.loss_and_grads(x, y)
    analytic = {(i, k): layer.grads[k].copy() for i, layer in enumerate(net.layers) for k in layer.params}
    worst = 0.0
    for (i, k), grad in analytic.items():
        p = net.layers[i].params[k]
        for flat in rng.choice(p.size, size=min(samples, p.size), replace=False):
            idx = np.unravel_index(flat, p.shape)
            old = p[idx]
            p[idx] = old + eps
            plus = net.loss.forward(net.forward(x, training=True), y)
            p[idx] = old - eps
            minus = net.loss.forward(net.forward(x, training=True), y)
            p[idx] = old
            numeric = (plus - minus) / (2 * eps)
            a = grad[idx]
            worst = max(worst, abs(a - numeric) / max(abs(a) + abs(numeric), 1e-12))
    return worst
