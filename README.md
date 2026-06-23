# GradientAscentML
<div align="center">

![Static Badge](https://img.shields.io/badge/github-repo-blue?logo=github)
![Static Badge](https://img.shields.io/badge/version-1.0.0-green)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

</div>

<br>
<p align= "center">
A project to explore how to forget.
</p>

## Mathematical Definitions

Given a dataset $`X \in \mathbb{R}^{N \times D}`$ with targets $`y \in \mathbb{R}^{N \times 1}`$, and a model $`\mathcal{F}(x \mid \theta)`$, let $`S \subseteq [N]`$ index the data to forget. For the sake of brevity, also define $`\bar{S} = [N] \setminus S`$ as the data to retain and define:

```math
\begin{gathered}
X_{forget} = X_{S,\,:} \qquad y_{forget} = y_{S,\,:}\\[4pt]
X_{retain} = X_{\bar{S},\,:} \qquad y_{retain} = y_{\bar{S},\,:}
\end{gathered}
```

These have dimensions:

```math
X_{forget} \in \mathbb{R}^{|S| \times D}, \quad y_{forget} \in \mathbb{R}^{|S| \times 1}, \quad X_{retain} \in \mathbb{R}^{|\bar{S}| \times D}, \quad y_{retain} \in \mathbb{R}^{|\bar{S}| \times 1}
```

Let $`\theta`$ be the parameters fit on the full dataset, and $`\theta_{retain}`$ the parameters fit on the retained data alone:

```math
\theta = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} L\big(y_i, \mathcal{F}(x_i \mid \theta)\big)
```

```math
\theta_{retain} = \arg\min_{\theta} \frac{1}{|\bar{S}|} \sum_{i \in \bar{S}} L\big(y_i, \mathcal{F}(x_i \mid \theta)\big)
```

where $`L`$ is the loss function.

**Problem.** Recover $`\theta_{retain}`$ using only $`\theta`$, $`\mathcal{F}`$, $`N`$, $`|S|`$, $`|\bar{S}|`$ and the forget set $`(X_{forget}, y_{forget})`$ — that is, without access to the retained data $`(X_{retain}, y_{retain})`$ or retraining from scratch.


## Forgetting in Standard Scaling
> [!IMPORTANT]
> Central idea: Recompute mean and standard deviations

Let $`\mu_i`$, $`\sigma_i`$ denote the mean and standard deviation used for model $`\mathcal{F}(x \mid \theta)`$ for feature $`i \in D`$ respectively, then:

```math
(\mu_{retain})_i = \frac{N \cdot \mu - \sum_{j \in [|S|]}(X_{forget})_{j, i}}{|\bar{S}|}
```

```math
(\sigma^2_{retain})_i = \frac{N \cdot (\sigma_i^2 + \mu_i^2) - \sum_{j \in [|S|]}(X_{forget})_{j, i}^2}{|\bar{S}|} - (\mu_{retain})_i^2
```

For future use also define:

```math
\mu = \begin{bmatrix} \mu_1 \\ \vdots \\ \mu_D \end{bmatrix} \in \mathbb{R}^D, \qquad
\sigma = \begin{bmatrix} \sigma_1 \\ \vdots \\ \sigma_D \end{bmatrix} \in \mathbb{R}^D, \qquad
\mu_{retain} = \begin{bmatrix} (\mu_{retain})_1 \\ \vdots \\ (\mu_{retain})_D \end{bmatrix} \in \mathbb{R}^D, \qquad
\sigma_{retain} = \begin{bmatrix} (\sigma_{retain})_1 \\ \vdots \\ (\sigma_{retain})_D \end{bmatrix} \in \mathbb{R}^D
```

## Forgetting in Simple Linear Regression
> [!IMPORTANT]
> Central idea: Recompute the Simple Linear Regression weights

The ordinary least squares solution of the simple linear regression on the design matrix $`\tilde{X} = [\,\mathbf{1} \mid X\,]`$ and $`y`$ is as follows:

```math
\theta = (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^Ty = G^{-1}M
```

where we call $`G`$ the Gram matrix and $`M`$ the Moment matrix. Then with $`\tilde{X}_{forget} = [\,\mathbf{1} \mid X_{forget}\,]`$, we find:

```math
G_{retain} = G - (\tilde{X}_{forget})^T\tilde{X}_{forget}
```

```math
M_{retain} = M - (\tilde{X}_{forget})^Ty_{forget}
```

```math
\theta_{retain} = G_{retain}^{-1}M_{retain}
```

> [!WARNING]
> When using forgetting in Simple Linear Regression with forgetting in Standard Scaling, then weights are not equivalent due to coordinate mapping differences. This must be corrected by rebasing the weights post-hoc.

For this, define the change of basis matrix:

```math
B = \begin{bmatrix}
1 & \boldsymbol{\tau} \\
\mathbf{0} & D
\end{bmatrix}, \quad \text{where} \quad \boldsymbol{\tau} = (\mu - \mu_{retain}) \oslash \sigma_{retain}, \qquad D = \text{diag}(\sigma \oslash \sigma_{retain})
```

Then we find that:

```math
G_{rebase} = B^TG_{retain}B
```

```math
M_{rebase} = B^TM_{retain}
```

```math
\theta_{rebase} = G_{rebase}^{-1}M_{rebase}
```

## Forgetting in Simple Logistic Regression
> [!IMPORTANT]
> Central idea: Recompute the Simple Logistic Regression weights

Contrary to simple linear regression, simple logistic regression has no closed-form solution, hence we must rely on approximations. We start by defining the Hessian on the design matrix $`\tilde{X} = [\,\mathbf{1} \mid X\,]`$ and parameters $`\theta`$:

```math
H = \tilde{X}^T \text{diag}(p_i(1-p_i))\tilde{X}, \qquad p_i = \tilde{X}_i^T \theta
```

Also define the Hessian for forgetting as:

```math
H_{forget} = \tilde{X}_{forget}^T \text{diag}\big((p_{forget})_i(1-(p_{forget})_i)\big)\tilde{X}_{forget}, \qquad (p_{forget})_i = (\tilde{X}_{forget})_i^T \theta
```

Then define the gradient of forgetting as:

```math
g_{forget} = \tilde{X}_{forget}^T (p_{forget} - y_{forget}), \qquad p_{forget} = \begin{bmatrix} (p_{forget})_1 \\ \vdots \\ (p_{forget})_{|S|} \end{bmatrix} \in \mathbb{R}^{|S|}
```

Then:

```math
H_{retain} = H - H_{forget}
```

```math
\theta_{retain} \approx \theta + H_{retain}^{-1}g_{forget}
```

To improve approximations, we can batch rows of the forget set and iteratively update the weights over these batches.


> [!WARNING]
> When using forgetting in Simple Logistic Regression with forgetting in Standard Scaling, then weights are not equivalent due to coordinate mapping differences. This must be corrected by rebasing the weights post-hoc.

For this, define the change of basis matrix (equivalent as in linear regression):

```math
B = \begin{bmatrix}
1 & \boldsymbol{\tau} \\
\mathbf{0} & D
\end{bmatrix}, \quad \text{where} \quad \boldsymbol{\tau} = (\mu - \mu_{retain}) \oslash \sigma_{retain}, \qquad D = \text{diag}(\sigma \oslash \sigma_{retain})
```

Then we find that:

```math
H_{rebase} = B^TH_{retain}B
```

```math
\theta_{rebase} = B^{-1}\theta_{retain}
```

## Credits
[@Cistaroth](https://github.com/Cistaroth)