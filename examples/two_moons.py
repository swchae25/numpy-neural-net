"""Train a small network on the two-moons data set and plot its decision boundary."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from nnumpy import Adam, Dense, Network, SoftmaxCrossEntropy, Tanh

rng = np.random.default_rng(42)
x, y = make_moons(n_samples=600, noise=0.25, random_state=42)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)

net = Network([Dense(2, 32, rng=rng), Tanh(), Dense(32, 32, rng=rng), Tanh(), Dense(32, 2, rng=rng)],
              SoftmaxCrossEntropy())
net.fit(x_tr, y_tr, Adam(lr=0.01, weight_decay=1e-4), epochs=400, batch_size=32,
        x_val=x_va, y_val=y_va, patience=40, rng=rng)
acc = net.accuracy(x_te, y_te)
print(f"two moons: test accuracy {acc:.3f} after {len(net.history['train_loss'])} epochs")

gx, gy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
logits = net.forward(np.c_[gx.ravel(), gy.ravel()])
p = np.exp(logits - logits.max(1, keepdims=True))
p = (p / p.sum(1, keepdims=True))[:, 1].reshape(gx.shape)

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
a.contourf(gx, gy, p, levels=20, cmap="RdBu_r", alpha=0.75)
a.contour(gx, gy, p, levels=[0.5], colors="k", linewidths=1.5)
a.scatter(*x_te.T, c=y_te, cmap="RdBu_r", edgecolors="k", s=22)
a.set_title(f"Decision boundary (test accuracy {acc:.1%})")
b.plot(net.history["train_loss"], label="train")
b.plot(net.history["val_loss"], label="validation")
b.set_xlabel("epoch"), b.set_ylabel("cross-entropy"), b.set_title("Learning curves (early stopping)")
b.legend()
fig.tight_layout()
fig.savefig(Path(__file__).parents[1] / "docs" / "two_moons.png", dpi=120)
