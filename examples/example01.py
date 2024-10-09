"""Plot Template Estimator.

===========================
Plotting Template Estimator
===========================

An example plot for using BDE estimator.
"""

import numpy as np
from matplotlib import pyplot as plt
from optax import adam

from bde.ml.loss import GaussianNLLLoss
from bde.ml.models import BDEEstimator, FullyConnectedModule

est = BDEEstimator(
    model_class=FullyConnectedModule,
    model_kwargs={
        "n_output_params": 2,
        "layer_sizes": [12, 12],
    },  # No hidden layers by default
    n_chains=20,  # 1 by default
    n_samples=20,  # 1 by default
    chain_len=500,  # 1 by default
    warmup=100,  # 1 by default
    optimizer_class=adam,
    optimizer_kwargs={
        "learning_rate": 1e-3,
    },
    loss=GaussianNLLLoss(),
    batch_size=128,  # 1 by default
    epochs=120,  # 1 by default
    metrics=None,
    validation_size=None,
    seed=42,
)

x = np.random.rand(256, 1).astype(np.float32)
x.sort()
y = x + 0.2
est = est.fit(x, y[..., 0])
y_pred = est.predict(x)

plt.plot(x, y, "k", alpha=0.4, label="True")
plt.scatter(x, y_pred, c="crimson", label="Prediction")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
