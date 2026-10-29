"""Classify 8x8 handwritten digits and compare with scikit-learn's MLPClassifier."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from nnumpy import Adam, Dense, Network, ReLU, SoftmaxCrossEntropy

rng = np.random.default_rng(7)
x, y = load_digits(return_X_y=True)
x = x / 16.0
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=7, stratify=y)
x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.15, random_state=7, stratify=y_tr)

net = Network([Dense(64, 128, rng=rng), ReLU(), Dense(128, 64, rng=rng), ReLU(), Dense(64, 10, rng=rng)],
              SoftmaxCrossEntropy())
net.fit(x_tr, y_tr, Adam(lr=1e-3, weight_decay=1e-4), epochs=300, batch_size=64,
        x_val=x_va, y_val=y_va, patience=25, rng=rng)
ours = net.accuracy(x_te, y_te)

ref = MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=1e-3, alpha=1e-4, batch_size=64,
                    max_iter=300, early_stopping=True, n_iter_no_change=25, random_state=7)
ref.fit(np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))
theirs = ref.score(x_te, y_te)
print(f"digits: nnumpy {ours:.3f} | scikit-learn MLPClassifier {theirs:.3f} (test accuracy)")

cm = confusion_matrix(y_te, net.predict(x_te))
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(10)), ax.set_yticks(range(10))
ax.set_xlabel("predicted"), ax.set_ylabel("true")
ax.set_title(f"Digits test set: accuracy {ours:.1%}")
fig.tight_layout()
fig.savefig(Path(__file__).parents[1] / "docs" / "digits_confusion.png", dpi=120)
