"""Plot Template Estimator.

===========================
Plotting Template Estimator
===========================

A simple example of using a BDE estimator.
"""

import numpy as np
from matplotlib import pyplot as plt
from optax import adam

from bde.ml.loss import GaussianNLLLoss
from bde.ml.models import BDEEstimator, FullyConnectedModule

# %%
# The following data will be used in this example

n_train = 256
x = np.random.rand(n_train * 2).astype(np.float32).reshape(-1, 1)
x_train = np.sort(x[:n_train], axis=0)
x_test = np.sort(x[n_train:], axis=0)
x = np.sort(x, axis=0)


def find_y(x, noise_level=None):
    r"""Calculate target value."""
    y = (5 * x + 1) ** 2
    y += 2
    y = y[..., 0]
    if noise_level is None:
        return y
    return y + np.random.normal(loc=0.0, scale=noise_level, size=x.shape[0])


y_train = find_y(x_train, noise_level=2.0)
y_test = find_y(x_test, noise_level=2.0)
y = find_y(x)

# %%
# We'll set up the following BDE estimator to solve this regression task

est = BDEEstimator(
    model_class=FullyConnectedModule,
    model_kwargs={
        "n_output_params": 2,
        "layer_sizes": [12, 12],
        "do_final_activation": False,
    },  # No hidden layers by default
    n_chains=12,  # 1 by default
    n_samples=30,  # 1 by default
    chain_len=200,  # 1 by default
    warmup=60,  # 1 by default
    optimizer_class=adam,
    optimizer_kwargs={
        "learning_rate": 1e-3,
    },
    loss=GaussianNLLLoss(),
    batch_size=128,  # 1 by default
    epochs=60,  # 1 by default
    metrics=None,
    validation_size=None,
    seed=42,
)

est = est.fit(x_train, y_train)
y_pred = est.predict(x_test)

# %%
# And examine the results

fig = plt.figure()
plt.scatter(x_test, y_pred, c="crimson", label="Prediction")
plt.scatter(x_test, y_test, c="green", label="True", alpha=0.2)
plt.plot(x, y, "k", alpha=0.4, label="Trend")
plt.title("Demonstrating value prediction")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %%
# Alternatively, the model could be used to predict the values with a
# confidence interval

y_pred, low, up = [
    np.ravel(v) for v in est.predict_with_credibility_eti(x_test, 0.95)
]

fig = plt.figure()
plt.fill_between(
    np.ravel(x_test),
    y1=low,
    y2=up,
    alpha=.5,
    linewidth=0,
    facecolor="crimson",
)
plt.plot(x_test, y_pred, c="crimson", label="Prediction")
plt.scatter(x_test, y_test, c="green", label="True", alpha=0.2)
plt.plot(x, y, "k", alpha=0.4, label="Trend")
plt.title("Demonstrating confidence interval prediction (95%)")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
