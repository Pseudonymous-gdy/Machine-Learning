
**Document Class** #Clustering  #DSAA2011 

**Table of Contents:**
- [[#Intro: How do we fit a Gaussian Distribution & Latent Variables|Intro: How do we fit a Gaussian Distribution & Latent Variables]]
- [[#Model Formulation|Model Formulation]]
- [[#Optimization Method: Expectation Maximization (EM)|Optimization Method: Expectation Maximization (EM)]]
	- [[#Optimization Method: Expectation Maximization (EM)#Algorithm|Algorithm]]
	- [[#Optimization Method: Expectation Maximization (EM)#Interpretation|Interpretation]]
- [[#Selection of K|Selection of K]]
- [[#Pros & Cons|Pros & Cons]]

**Soft Clustering Method by Implementing Gaussian Mixtues.**

---
## Intro: How do we fit a Gaussian Distribution & Latent Variables

In [[Clustering#Soft Clustering]], gaussian mixture model is a "universal approximator" of densities (probability density function), meaning that any smooth density can be approximated with any specific nonzero amount of error by a Gaussian Mixture Model with enough components.

For `univariate Gaussian Distribution`, we obtain the likelihood function as$$p(\mathcal{D}|\mu,\sigma^2)=\prod^n_{i=1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$With a solution conducted by maximizing `Log-likelihood` in [[Maximum Likelihood Estimator|MLE]] Skeleton $$\begin{aligned}\mathcal{L}&=\sum^n_{i=1}\left(-\frac{1}{2}\log(2\pi\sigma^2)-\frac{(x_i-\mu)^2}{2\sigma^2}\right)\\
\frac{\partial\mathcal{L}}{\partial \mu}&=-\sum^n_{i=1}\frac{2(x-\mu)}{2\sigma^2}=0\Rightarrow\mu_{ML}=\frac{\sum^n_{i=1}x_i}{n}\\
\frac{\partial\mathcal{L}}{\partial\sigma}&=-\frac{n}{\sigma}+\sum^n_{i=1}\frac{(x_i-\mu)^2}{\sigma^3}=0\Rightarrow\sigma^2_{ML}=\sum^n_{i=1}\frac{(x_i-\mu)^2}{n}\end{aligned}$$
Similarly, for multi-variate Gaussian Distribution with a likelihood function of$$p(\mathcal{D}|\mu,\Sigma)=\prod^n_{i=1}\frac{1}{\sqrt{(2\pi)^D|\Sigma|}}\exp\left(-\frac{(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)}{2}\right)$$Where $\mu$ is the mean parameter of the vectors, and $\Sigma$ as the Covariance matrix of all $n$ random variables. By maximizing the Log-likelihood, we have $$\begin{aligned}
\mathcal{L}&=\sum^n_{i=1}\left(-\frac{D}{2}\log(2\pi)-\frac{1}{2}\log|\Sigma|-\frac{(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)}{2}\right)\\
\frac{\partial\mathcal{L}}{\partial\mu}&=\sum^n_{i=1}(-1)\cdot\Sigma^{-1}(x_i-\mu)=-\Sigma^{-1}\sum^n_{i=1}(x_i-\mu)=0\\&\Rightarrow\mu_{ML}=\sum^n_{i=1}x_i/n\\
\frac{\partial\mathcal{L}}{\partial\Sigma}&=-\frac{n}{2}\left(\Sigma^{-1}\right)^T+\frac{1}{2}\sum^n_{i=1}\Sigma^{-1}(x_i-\mu)(x_i-\mu)^T\Sigma^{-1}\\&=\frac{\Sigma^{-1}}{2}\left(\sum^n_{i=1}(x_i-\mu)(x_i-\mu)^T-n\Sigma\right)\Sigma^{-1}=0\\
&\Rightarrow\Sigma=\frac{1}{n}\sum^n_{i=1}(x_i-\mu)(x_i-\mu)^T\end{aligned}$$

`Why not single Gaussian`? For its failure to capture complex data, i.e., data with sub distributions. To solve this, we introduce `latent variable` (the variables you can never observe, but can only be inferred indirectly from other observable variables) $z_i$ to model such sub distributions. In brief, we state that an observation $x_i$ is caused by some underlying latent variable satisfying:
- $p(x,z|\theta)=p(x|z,\theta)\cdot p(z|\theta)$
- $p(x|\theta)=\int p(x,z|\theta)dx=\sum_zp(x,z|\theta)$
- (`Basics of Bayesian`) $P(A=a|B=b)=\frac{P(A=a,B=b)}{\sum_{x\in A}P(A=x,B=b)}$
Where $\theta$ indicates the model parameters. Then, by combining the terms:
- $p(x_i|z_i=k,\theta)=\mathcal{N}(x|\mu_k,\Sigma_k)$
- $p(z_i=k|\theta)=\pi_k$
- $p(x_i|\theta)=\Sigma_zp(x_i,z|\theta)=\sum^K_{k=1}\pi_k\mathcal{N}(x_i|\mu_k,\Sigma_k)$
We get the following formulation.

## Model Formulation

The probability is the weighted sum of a number of pdfs where the weights are determined by a distribution$$\begin{aligned}p(x)&=\pi_{1} f_{1}(x)+\pi_{2} f_{2}(x)+...+\pi_{K} f_{K}(x)\\\sum_{k=1}^{K} \pi_{k}&=1\end{aligned}$$Where for GMM, $f_i(x)=\mathcal{N}(x|\mu_i,\Sigma_i)$. Then by log-likelihood, we have
$$\begin{aligned}
p(X|\theta)&=\prod^n_{i=1}\sum^K_{k=1}\pi_k\frac{1}{\sqrt{(2\pi)^D|\Sigma|}}\exp\left(-\frac{(x_i-\mu_k)^T\Sigma^{-1}(x_i-\mu_k)}{2}\right)\\
\mathcal{L}&=\sum^n_{i=1}\log\left(\sum^K_{k=1}\pi_k\frac{1}{\sqrt{(2\pi)^D|\Sigma|}}\exp\left(-\frac{(x_i-\mu_k)^T\Sigma^{-1}(x_i-\mu_k)}{2}\right)\right)\\
\frac{\partial \mathcal{L}}{\partial\mu_k}&=\sum^n_{i=1}\frac{\pi_k\frac{1}{\sqrt{(2\pi)^D|\Sigma|}}\exp\left(-\frac{(x_i-\mu_k)^T\Sigma^{-1}(x_i-\mu_k)}{2}\right)}{\sum^K_{k=1}\pi_k\frac{1}{\sqrt{(2\pi)^D|\Sigma|}}\exp\left(-\frac{(x_i-\mu_k)^T\Sigma^{-1}(x_i-\mu_k)}{2}\right)}\times\Sigma^{-1}(x_i-\mu_k)=0
\end{aligned}$$
Which is our target though complicated.

## Optimization Method: Expectation Maximization (EM)

### Algorithm
The goal of EM is to iteratively estimate $\theta=\{\pi_{i}, \mu_{i}, \sum _{i}\}_{i=1}^{K}$ using unlabeled data. The algorithm is listed as below:
- Init: Randomly initialize $\theta$
- **E-Step:** Compute **responsibilities**$$r_{i k}=\frac{\pi_{k} \mathcal{N}\left(x_{i} | \mu_{k}, \sum_{k}\right)}{\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(x_{i} | \mu_{k}, \sum_{k}\right)}, \forall i, k$$
- **M-Step:** Update parameters$$\mu_{k}=\frac{\sum_{i} r_{i k} x_{i}}{\sum_{i} r_{i k}}, \sigma_{k}^{2}=\frac{\sum_{i} r_{i k}\left(x_{i}-\mu_{k}\right)^{2}}{\sum_{i} r_{i k}}, \pi_{k}=\frac{\sum_{i} r_{i k}}{n}$$
- Check: Stop if log-likelihood converges or max iterations reached

### Interpretation
To understand this algorithm, think about MLE:$$\theta_{MLE}=\arg\max\log p(X|\theta)=\arg\max\log\sum_Zp(X,Z|\theta)$$Where $Z$ is a latent variable and cannot be observed, which prevents a simple analytical solution. However, we can obtain its posterior distribution given $\theta^{\text{old}}$ using Bayes Theorem:$$p(Z|X,\theta^\text{old})\propto p(Z|\theta^\text{old})p(X|Z,\theta^\text{old})=\frac{p(Z|\theta^\text{old})p(X|Z,\theta^\text{old})}{\sum_Zp(Z|\theta^\text{old})p(X|Z,\theta^\text{old})}$$
To get the posterior, we have E-step and M-step for it. First, we could get a "distance" between the estimated probability and the expectation of the complete-data $L(\theta,q)$ by$$\begin{aligned} \log p(X | \theta) & =\log \sum_{Z} p(X, Z | \theta) \\ & =\log \sum_{Z} q(Z) \frac{p(X, Z | \theta)}{q(Z)} \\ & =\log \mathbb{E}_{q(Z)} \frac{p(X, Z | \theta)}{q(Z)} \\ & \geq \mathbb{E}_{q(Z)} \log \frac{p(X, Z | \theta)}{q(Z)} \quad\text{By Jensen's Inequality:} \mathbb{E}[f(X)]\leq f(\mathbb{E}[X])\text{ if }f \text{ concave}\\ & =L(\theta, q) \end{aligned}$$Where $q$ is the probability density funtion of $Z$, which is not observable; and $L$ is the lower bound function of $\theta$ and $q$.

Then we get$$\begin{aligned} \log p(X | \theta)-L(\theta, q) & =\log p(X | \theta)-\mathbb{E}_{q(Z)} \log \frac{p(X, Z | \theta)}{q(Z)} \\ & =\mathbb{E}_{q(Z)} \log \frac{p(X | \theta) q(Z)}{p(X, Z | \theta)} \\ & =\mathbb{E}_{q(Z)} \log \frac{q(Z)}{p(Z | X, \theta)} \\ & =\text{KL}(q(Z), p(Z | X, \theta))\quad (\text{Kullback-Leibler divergence}) \end{aligned}$$So we have $L(\theta, q)=\log p(X | \theta)-\text{KL}(q(Z), p(Z | X, \theta))$ which is `Evidence Lower BOund (ELBO)`.

Given the result, we analyze the intermediate steps.
1. E-step
	In the E-step, $L(\theta,q)$ is maximized with respect to $q$ while $\theta$ is held fixed, which is given by$$q^\text{new}=\arg\max_qL(\theta^\text{old},q)=\arg\min_q\text{KL}(q(Z),p(Z|X,\theta))$$Which we therefore set $q^\text{new}(\cdot)=p(\cdot|X,\theta)$, which is done by responsibility calculation. Therefore, as the approximation of posterior distribution, $r_{ik}$ is calculated by$$\begin{aligned}r_{ik}&=p(Z=k|x_i,\theta)=\frac{p(Z=k,x_i|\theta)}{p(x_i|\theta)}\\&=\frac{\pi_{k} \mathcal{N}\left(x_{i} | \mu_{k}, \sum_{k}\right)}{\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(x_{i} | \mu_{k}, \sum_{k}\right)}\end{aligned}$$P.S.: When the posterior is still hard to obtain, we can use other distribution to “approximate” it, which is Variational Inference. 

2. M-step
	In the M-step, $L(\theta,q)$ is maximized with respect to $\theta$ while $q$ is held fixed, which is given by$$\theta^\text{new}=\arg\max_\theta L(\theta,q^\text{new})=\arg\max_\theta\log\frac{p(X,Z|\theta)}{q^\text{new}(Z)}=\arg\max_\theta\log p(X,Z|\theta)$$
	Since we've got the term for $p(X,Z|\theta)$, we could take derivatives with respect to $\theta$ and get $\mu_k,\sigma^2_k,\pi_k$.


## Selection of K

Usually, we do not know the optimal number of components in prior, but we can get a hint when plotting the lower bound L vs. the number of mixture components. Here we'll get a plot of relations, which indicates certain information in the "elbow" points. There, an increase in $K$ leads to insignificant increase in the lower bound.

## Pros & Cons
1. Pros
	- Easy to implement for uni/multivariate data
	- EM converges to a (local) minimum
	- Log-likelihood decreases each iteration
	- Faster and more stable than gradient descent
	- Handles covariance constraints well
2. Cons
	- Sensitive to parameter initialization
	- Risk of singularities (where certain $\sigma^2\to0$, with larger responsibilities, thereby smaller $\sigma^2$, and an iterated cycle within it)
	- May overfit without validation
