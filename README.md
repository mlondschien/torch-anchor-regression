# torch-anchor-regression

Implementation of CVaR anchor regression in PyTorch.

Some details:

- `data.StratifiedSampler` yields batches with predefined number of environments per batch and approx. observations per environment. Draws are without replacement, i.e., the sampler yields each observation once per epoch. Thus, also, observations and environments are correctly weighted. Increase the number of environments to reduce the variance due to the CVaR tail selection. Increase the number of observations per environment to reduce variance of the estimation of projected residuals (group-wise means).
- `loss.AnchorLoss(alpha, gamma, tau, debias=True)` yields Rockafellar-Uryasev formulation $`\min_t \sum_e w_e \hat\sigma^2_e + \gamma \left( t + \frac{1}{\alpha} \sum_e w_e (\hat\mu_e^2 - \hat\sigma^2_e / n_e - t)_+ \right)`$ of $`\sum_e w_e \hat\sigma^2_e + \gamma \mathrm{CVaR}_\alpha( \hat\mu_e^2 - \hat\sigma^2_e / n_e )`$ with residuals $`r_i`$, $`w_e = n_e / n`$, $`\hat\sigma^2_e = \frac{1}{n_e - 1} \sum_{i \in e} (r_i - \hat\mu_e)^2`$ and $`\hat\mu_e = \frac{1}{n_e} \sum_{i \in e} r_i`$ and approximating the ReLU with softplus $`\tau \log(1 + e^{x / \tau})`$. `AnchorLoss().threshold` is a learned parameter with preferably no weight decay.
