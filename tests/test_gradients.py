import numpy as np
import pytest

from nnumpy import (Adam, BinaryCrossEntropy, Dense, MeanSquaredError, Network, ReLU, SGD, Sigmoid,
                    SoftmaxCrossEntropy, Tanh, gradient_check)


def make_net(act, loss, n_out, rng):
    return Network([Dense(4, 7, rng=rng), act(), Dense(7, 5, rng=rng), act(), Dense(5, n_out, rng=rng)], loss)


@pytest.mark.parametrize("act", [Tanh, Sigmoid, ReLU])
@pytest.mark.parametrize("loss,n_out,make_y", [
    (SoftmaxCrossEntropy(), 3, lambda r: r.integers(0, 3, 16)),
    (BinaryCrossEntropy(), 1, lambda r: r.integers(0, 2, (16, 1)).astype(float)),
    (MeanSquaredError(), 2, lambda r: r.normal(size=(16, 2))),
])
def test_backprop_matches_finite_differences(act, loss, n_out, make_y):
    rng = np.random.default_rng(1)
    net = make_net(act, loss, n_out, rng)
    x = rng.normal(size=(16, 4))
    assert gradient_check(net, x, make_y(rng)) < 1e-5  # typical tolerance for float64 central differences


def test_softmax_cross_entropy_is_stable_for_huge_logits():
    loss = SoftmaxCrossEntropy()
    value = loss.forward(np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.array([0, 1]))
    assert np.isfinite(value) and value < 1e-6


def test_bce_on_logits_matches_definition():
    z = np.array([[-3.0], [0.5], [4.0]])
    t = np.array([[0.0], [1.0], [1.0]])
    p = 1 / (1 + np.exp(-z))
    expected = -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()
    assert BinaryCrossEntropy().forward(z, t) == pytest.approx(expected)


@pytest.mark.parametrize("opt", [Adam(lr=0.05), SGD(lr=0.5, momentum=0.9)])
def test_learns_xor(opt):
    rng = np.random.default_rng(0)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    net = Network([Dense(2, 8, rng=rng), Tanh(), Dense(8, 2, rng=rng)], SoftmaxCrossEntropy())
    net.fit(x, y, opt, epochs=500, batch_size=4)
    assert net.accuracy(x, y) == 1.0


def test_early_stopping_restores_best_weights():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    net = Network([Dense(2, 16, rng=rng), ReLU(), Dense(16, 2, rng=rng)], SoftmaxCrossEntropy())
    hist = net.fit(x[:150], y[:150], Adam(lr=0.01), epochs=300, x_val=x[150:], y_val=y[150:], patience=20)
    final_val = net.loss.forward(net.forward(x[150:]), y[150:])
    assert final_val == pytest.approx(min(hist["val_loss"]), rel=1e-9)
