**Document Class** #DSAA2011 #Mathematics

**Table of Contents:**
- [[#Methodology|Methodology]]
- [[#Example|Example]]
- [[#Pros & Cons|Pros & Cons]]

---
## Methodology

Basically, we conduct such methodology for estimating parameters based on a prior assumption and observed data. The procedure goes as follows:
- Assume the distribution based on prior/educated guesses.
- Calculate the likelihood function $L(x;\theta)$, where $x$ is the observed data, and $\theta$ is the parameters.
- Find the value of parameters when maximizing the likelihood function.

## Example

Suppose we're going to estimate the parameter of $p$ of the Bernoulli Distribution based on the observed data $\mathbf{x}=(x_1,x_2,\cdots,x_n)$ with $x_1,x_2,\cdots,x_n$ are $i.i.d$s following a Bernoulli Distribution.

Then we have
$$\begin{aligned}
L(\mathbf{x};p)&=\prod^n_{i=1}p^{\mathbb{1}\{x_i=1\}}(1-p)^{\mathbb{1}\{x_i=0\}}
\\
\Rightarrow\text{maximize}\quad \log(L(\mathbf{x};p))&=\sum^n_{i=1}\mathbb{1}\{x_i=1\}\log p+\mathbb{1}\{x_i=0\}\log(1-p)\\
\Rightarrow\frac{\partial \log L}{\partial p}&=\frac{n_+}{p}-\frac{n_-}{1-p}=0\\
\Leftrightarrow p&=\frac{n_+}{n}
\end{aligned}$$

## Pros & Cons

Usually, we explore the MLE due to several `nice properties`:
- Efficiency: MLE is efficient, achieving the lowest possible variance among all unbiased estimators.
- Asymptotic Normality: $\sqrt{n}(\hat{\theta}_{ML}-\theta)\rightarrow^d\mathcal{N}(0,I(\theta)^{-1})$
- Consistency, sample size increases to infinity, the estimator will converge to the true parameter value
- Unbiasedness: MLE is not necessarily unbiased, but in certain cases, it can be unbiased.
	Counterexample: $X_1,X_2,...,X_m\sim\exp{(\theta)}$, then:$$\begin{aligned}
	\mathbb{E}[\hat{\theta}]&=m\cdot\mathbb{E}\bigl[\frac{1}{\sum X_i}\bigr]\\
	&=m\cdot\int^{+\infty}_{0}\frac{1}{x}\cdot{gamma(x)}dx\ \text{(Sum of m i.i.d $\exp{\theta}$ follows Gamma($m,\theta$))}\\
	&=\frac{m\theta}{m-1}\neq\theta
	\end{aligned}$$
However, MLE is flawed for:
- Model Sensitivity: model assumptions (affect likelihood functions)