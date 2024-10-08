# Bayesian Deep Ensembles (BDE)

## Introduction

This repo was created as a course project at the University of Munich
(LMU). It implements a bayesian deep ensemble of fully connected 
networks for use with tabular data.
The following links contain the [background paper](https://arxiv.org/abs/2402.01484 )
and the [repo](https://github.com/EmanuelSommer/bnn_connecting_the_dots) corresponding to the background paper. 
The package is compatible with [Jax](https://jax.readthedocs.io/en/latest/quickstart.html) and [sklearn](https://scikit-learn.org/stable/index.html).


## Development Setup

- Install pre-commit hooks with `pre-commit install`
- Run tests using `pytest`

## Background
Bayesian Neural Networks provide a principled approach to deep learning 
which allows for uncertainty quantification. Compared to traditional
statistical methods which treat model parameters as unknown, but fixed
values, Bayesian methods treat model parameters as random
variables. Hence, we have to specify a prior distribution over those
parameters which can be interpreted as prior knowledge. Given data,
we can update the beliefs about the parameters and calculate credible
intervals for the parameters and predictions. A credible interval in 
Bayesian statistics defines the range for which the parameter or prediction is 
believed to fall into with a specified probability based on its posterior distribution. 

However, while potentially rewarding for its predictive capabilities and uncertainty
measurements, Bayesian optimization can be challenging and resource intensive due to 
usually strongly multimodal posterior landscapes.
([Izmailov et al., 2021](https://proceedings.mlr.press/v139/izmailov21a.html))
To alleviate that issue, this package uses an ensemble of networks sampled from different Markov
Chains to better capture the posterior density and [Jax](https://jax.readthedocs.io/en/latest/quickstart.html) for better computational efficiency.

## The Procedure
Assumptions: assume an independent distribution of model parameters
1. Define a fully connected neural network structure where each output value corresponds to a parametrization of a distribution. 
   In the case of a Gaussian distribution (currently the only supported option), each output value corresponds to 2 predictions: mean $\mu$ and standard deviation $\sigma$, and the output layer for a network with N predicted values should look as follows: $(\mu_1, \mu_2, ..., \mu_N, \sigma_1, \sigma_2, ..., \sigma_N)$.
2. Train n neural networks in parallel using a
negative log-likelihood loss function to obtain $\mu$
and $log(\sigma)$ or $log(b)$ for Laplace, respectively. 
3. Specify a prior distribution over the model weights
4. Calculate the posterior probability of the weights
5. Use a sampler with burn-in period to sample new trained networks, 
i.e. sets of weights, in parallel from the posterior distribution
6. Use the obtained networks to predict the data
7. From the posterior predictive distribution, obtain mean estimates
and credible intervals 

The fully connected Bayesian networks are individually trained using 
Negative Losslikelihood Loss (NLL) with either Gaussian or Laplace Priors, i.e.
$$ 
\text{NLL}_{\text{Gaussian}}(y, \mu, \log \sigma) = \log( \sigma ) + \frac{(y - \mu)^2}{2 \sigma^2} + \frac{1}{2} \log(2 \pi)
$$
or
$$
\text{NLL}_{\text{Laplace}}(y, \mu, b) = \log(2b) + \frac{|y - \mu|}{b}
.$$

Given data $\mathcal{D}$, we can then calculate the posterior distribution of the 
parameters $\theta$, our network weights, as 
$p(\theta|\mathcal{D}) = \frac{p(\mathcal{D}|\theta)p(\theta)}{p(\mathcal{D})}$.
Using that posterior, for a new data point (x*, y*), we can then define the posterior 
predictive density (PPD) over the labels y as 
$$
p(y^* | x^*, \mathcal{D}) = \int_{\Theta} p(y^* | x^*, \theta) p(\theta | \mathcal{D}) \, d\theta.
$$
The PPD captures the uncertainty about the model, but usually has to be approximated as
$$
p(y^* | x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^{S} p(y^* | x^*, \theta^{(s)})
$$
through Monte Carlo sampling S samples from a Markov Chain that converged 
to the posterior density $p(\theta|\mathcal{D})$ such that
$\theta^{(s)} \sim p(\theta | \mathcal{D})$. 

Eq Test
1. The posterior predictive distribution (PPD):
   p(y* | x*, D) = ∫ p(y* | x*, θ) p(θ | D) dθ

2. Approximation of the PPD:
   p(y* | x*, D) ≈ (1/S) ∑[s=1 to S] p(y* | x*, θ^(s))

3. NLL_Gaussian(y, μ, log σ) = log(σ) + (y - μ)^2 / (2σ^2) + 1/2 log(2π)

4. NLL_Laplace(y, μ, b) = log(2b) + |y - μ| / b

5. $$p(y^* | x^*, \mathcal{D}) = \int_{\Theta} p(y^* | x^*, \theta) p(\theta | \mathcal{D}) \, d\theta.$$

6. text $`p(y^* | x^*, \mathcal{D}) = \int_{\Theta} p(y^* | x^*, \theta) p(\theta | \mathcal{D}) \, d\theta.`$

7.
```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```

## License

This project is licensed under the BSD 3-clause "New" or "Revised" license - see the [LICENSE](LICENSE) file for details.
