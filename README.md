[![Unit Tests](https://github.com/applied-ml-bde/bde/actions/workflows/python-app.yml/badge.svg)](https://github.com/applied-ml-bde/bde/actions/workflows/python-app.yml)
[![Linter](https://github.com/applied-ml-bde/bde/actions/workflows/lint.yml/badge.svg)](https://github.com/applied-ml-bde/bde/actions/workflows/lint.yml)
[![Documentation](https://github.com/applied-ml-bde/bde/actions/workflows/deploy-gh-pages.yml/badge.svg)](https://github.com/applied-ml-bde/bde/actions/workflows/deploy-gh-pages.yml)
# bde

## Introduction


This repo was created as a course project at the University of Munich
(LMU). It implements a bayesian deep ensemble of fully connected 
networks for use with tabular data.
The following links contain the [background paper](https://arxiv.org/abs/2402.01484 )
and the [repo](https://github.com/EmanuelSommer/bnn_connecting_the_dots) corresponding to the background paper. 
The package is compatible with [Jax](https://jax.readthedocs.io/en/latest/quickstart.html) and [sklearn](https://scikit-learn.org/stable/index.html).


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
from bde.ml.loss import GaussianNLLLoss
from optax import adam
from jax import numpy as jnp

est = BDEEstimator(
    model_class=FullyConnectedModule,
    model_kwargs={
        "n_output_params": 2,
        "layer_sizes": [10, 10],
    },  # No hidden layers by default
    n_chains=10,  # 1 by default
    n_samples=10,  # 1 by default
    chain_len=100,  # 1 by default
    warmup=100,  # 1 by default
    optimizer_class=adam,
    optimizer_kwargs={
        "learning_rate": 1e-3,
    },
    loss=GaussianNLLLoss(),
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

## Background
Bayesian Neural Networks provide a principled approach to deep learning 
which allows for uncertainty quantification. Compared to traditional
statistical methods which treat model parameters as unknown but fixed
values, Bayesian methods treat model parameters as random
variables. Hence, we have to specify a prior distribution over these
parameters which can be interpreted as prior knowledge. 
Given data, we can update the beliefs about the parameters and calculate credible
intervals for the parameters and predictions.
A credible interval in Bayesian statistics defines the range for which the
parameter or prediction is believed to fall into with a specified probability based
on its posterior distribution. 

However, while potentially rewarding for its predictive capabilities and uncertainty
measurements, Bayesian inference can be challenging and resource intensive due to 
usually strongly multimodal posterior landscapes.
([Izmailov et al., 2021](https://proceedings.mlr.press/v139/izmailov21a.html))
To alleviate that issue, this package uses an ensemble of networks sampled from
different Markov Chains to better capture the posterior density and 
[Jax](https://jax.readthedocs.io/en/latest/quickstart.html) for better computational
efficiency.

## The Procedure
Assumptions: assume an independent distribution of model parameters
1. Define a fully connected neural network structure where each output value
   corresponds to a parametrization of a distribution. 
   In the case of a Gaussian distribution (currently the only supported option),
   each output value corresponds to 2 predictions:
    - mean $\mu$ 
    - standard deviation $\sigma$. 
   Hence, the output layer for a network with N predicted values should look as 
   follows: $(\mu_1, \mu_2, ..., \mu_N, \sigma_1, \sigma_2, ..., \sigma_N$).
2. Train n neural networks in parallel using a
   negative log-likelihood loss function to obtain $\mu$
   and $\sigma$.
3. Specify a prior distribution over the model weights.
4. Calculate the unnormalized log posterior of the weights.
5. Use a sampler with burn-in period to sample new trained networks, 
   i.e. sets of weights, in parallel from the posterior distribution.
6. Use the obtained networks to predict the data.
7. From the posterior predictive distribution, obtain mean estimates
and credible intervals.

 <!--
The fully connected Bayesian networks are individually trained using 
Negative Losslikelihood Loss (NLL) with either Gaussian or Laplace Priors, i.e.
```math
\text{NLL}_{\text{Gaussian}}(y, \mu, \log \sigma) = \log( \sigma ) + \frac{(y - \mu)^2}{2 \sigma^2} + \frac{1}{2} \log(2 \pi)
```
or
```math
\text{NLL}_{\text{Laplace}}(y, \mu, b) = \log(2b) + \frac{|y - \mu|}{b}
.
```
-->
The fully connected Bayesian networks are individually trained using 
Negative Losslikelihood Loss (NLL) with Gaussian Priors, i.e.
```math
\text{NLL}_{\text{Gaussian}}(y, \mu, \log \sigma) = \log( \sigma ) + \frac{(y - \mu)^2}{2 \sigma^2} + \frac{1}{2} \log(2 \pi).
```

Given data $\mathcal{D}$, we can then calculate the posterior distribution of the 
parameters $\theta$, our network weights, as 
$p(\theta|\mathcal{D}) = \frac{p(\mathcal{D}|\theta)p(\theta)}{p(\mathcal{D})}$.
Using that posterior, for a new data point (x*, y*), we can then define the posterior 
predictive density (PPD) over the labels y as 
```math
p(y^* | x^*, \mathcal{D}) = \int_{\Theta} p(y^* | x^*, \theta) p(\theta | \mathcal{D}) \, d\theta.
```
The PPD captures the uncertainty about the model, but usually has to be approximated as
$`
p(y^* | x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^{S} p(y^* | x^*, \theta^{(s)})
`$
through Monte Carlo sampling S samples from a Markov Chain that converged 
to the posterior density $p(\theta|\mathcal{D})$ such that
$\theta^{(s)} \sim p(\theta | \mathcal{D})$. 


## License

This project is licensed under the BSD 3-clause "New" or "Revised" license - see the [LICENSE](LICENSE) file for details.
