[![Unit Tests](https://github.com/applied-ml-bde/bde/actions/workflows/python-app.yml/badge.svg)](https://github.com/applied-ml-bde/bde/actions/workflows/python-app.yml)
[![Linter](https://github.com/applied-ml-bde/bde/actions/workflows/lint.yml/badge.svg)](https://github.com/applied-ml-bde/bde/actions/workflows/lint.yml)
[![Documentation](https://github.com/applied-ml-bde/bde/actions/workflows/deploy-gh-pages.yml/badge.svg)](https://github.com/applied-ml-bde/bde/actions/workflows/deploy-gh-pages.yml)
# bde

## Introduction

The [background paper](https://arxiv.org/abs/2402.01484 )
and the [repo](https://github.com/EmanuelSommer/bnn_connecting_the_dots) corresponding to the paper.

## Development Setup

- Install [pixi](https://pixi.sh/latest/#installation) to manage development environments.
- To install all environments, run `pixi install -a --frozen` in the project directory.
- Install pre-commit hooks with `pre-commit install`.
- Run tests using `pytest` or `pixi run -e test test` from the project directory.
- For development, the interpreter in `bde/.pixi/envs/dev/` can be used.

## How to use

### Examples

[]("" "ADD: intro")

```python
from bde.ml.models import BDEEstimator
from jax import numpy as jnp

def_estimator = BDEEstimator()
x = jnp.arange(20, dtype=float).reshape(-1, 2)
y = x[..., -1]
def_estimator.fit(x, y)
y_pred = def_estimator.predict(x)
```

Due to the computational complexity of the models, most parameters were kept very small
for more efficient testing and experimentation.
In production environment most estimator parameters should be adjusted:

```python
from bde.ml.models import BDEEstimator, FullyConnectedModule
from bde.ml.loss import LogLikelihoodLoss
from optax import adam
from jax import numpy as jnp

est = BDEEstimator(
    model_class=FullyConnectedModule,
    model_kwargs={
        "n_output_params": 2,
        "layer_sizes": [10, 10],
    },  # No hidden layers by default
    n_chains=10,  # 1 by default
    n_init_runs=2,  # 1 by default
    chain_len=100,  # 1 by default
    warmup=100,  # 1 by default
    optimizer_class=adam,
    optimizer_kwargs={
        "learning_rate": 1e-3,
    },
    loss=LogLikelihoodLoss(),
    batch_size=2,  # 1 by default
    epochs=5,  # 1 by default
    metrics=None,
    validation_size=None,
    seed=42,
)

x = jnp.arange(20, dtype=float).reshape(-1, 2)
y = x[..., -1]
est.fit(x, y)
y_pred = est.predict(x)
```

Our estimator classes are compatible with `SKlearn` and can be used with their tools
for task such as hyperparameter optimization:

```python

```
[]("" "ADD: an example of using grid-search with `BDEEstimator`")

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
