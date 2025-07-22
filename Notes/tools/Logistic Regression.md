
# Motivation

Classifications based on linear regression are subject to problems:
- Noise: data not separable
- Mediocre generalization: finds a 'barely' separating solution
- Overtraining: test/held-out accuracy falls after rises (a kind of overfitting)

For non-separable case, it might be better to draw a `probabilistic decision` rather than a `deterministic decision`. And the problem becomes to get probabilistic decisions.

# Formula

For logistic function/sigmoid function:
$$
g(z)=\frac{1}{1+\exp(-z)}, x\in\mathbb{R}
$$
There are several good property:
- $g(z)\in(0,1)$, which is mapped into the range of probability measures.

Therefore, we interpret the logistic regression in such a way that *the **log-odds** of the predicted class probabilities should be a linear function of the inputs*:

$$\begin{aligned}
\log\frac{\Pr (y=1|\mathbf{x},\mathbf{\theta},\theta_0)}{\Pr (y=-1|\mathbf{x},\mathbf{\theta},\theta_0)}=\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0
\end{aligned}$$
Where we derive
$$
\Pr (y=1|\mathbf{x},\mathbf{\theta},\theta_0)=g(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0)=\frac{1}{1+\exp{\left(-\left\langle\mathbf{\theta},\mathbf{x}\right\rangle-\theta_0\right)}}
$$

A further investigation is included into [[#Q&A Why sigmoid function that changes the linear model into [0,1] could represent probability?]]

From above, we conclude that logistic regression is a generalized linear model because the class conditional probabilities are **a function (namely sigmoid) of a linear function** of the sample $\mathbf{x}$.

Specially, if sample $\mathbf{x}$ sits on the decision boundary where $\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0=0$, we conclude that $\Pr (y=1|\mathbf{x},\mathbf{\theta},\theta_0)=\Pr (y=0|\mathbf{x},\mathbf{\theta},\theta_0)=\frac{1}{2}$. In a plane of data points with features, the boundary between the two class is a hyperplane which, is distant from those who has high probability of being in one of the two classes.

And for further convenience, we declare a simplified way for the class probabilities.

$$
\Pr (y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)=g\left(y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)=\frac{1}{1+\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}}
$$

With the circumstance, we apply `MLE` to estimate parameters $(\mathbf{\theta},\theta_0)$.


# MLE & SGD Exploration

## MLE Exploration

Then we state the $\textcolor{blue}{\text{Likelihood}}$ of $(\mathbf{x}_t,y_t)$ given $(\mathbf{\theta},\theta_0)$ indicates the probability of seeing one sample $(\mathbf{x}_t,y_t)$ given that the current parameters are $(\mathbf{\theta},\theta_0)$ is
$$
\textcolor{blue}{L\left((\mathbf{\theta},\theta_0)|\mathbf{x}_t,y_t\right)}:=\Pr (y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)=g\left(y_t(\left\langle\mathbf{x}_t,\mathbf{\theta}\right\rangle)+\theta_0\right)
$$

While the $\textcolor{blue}{\text{Likelihood}}$ of the dataset $\mathcal{D}$ indicates the overall likelihood of seeing the dataset given the parameters
$$
L(\mathbf{\theta},\theta_0|\mathcal{D})=\prod^n_{t=1}L\left((\mathbf{\theta},\theta_0)|\mathbf{x}_t,y_t\right)=\prod^n_{t=1}\Pr (y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)
$$

Which is based on **independence** between samples, thus little correlation.

Then we explore the MLE due to several `nice properties`:
- consistent/unbiased with L2-norm (get the right parameter values in the limit of a large number of training examples).
- Asymptotically normal.
- Efficient in convergence rate.

And based on the `assumptions` of:
- Right model class is selected, which is logistic regression model
- Certain regularity conditions hold, which is L2-norm: $L(\mathbf{\theta},\theta_0|\mathcal{D})+\lambda\|\theta\|_2$

We conduct MLE estimation:

$$\begin{aligned}
\arg\max_{(\mathbf{\theta},\theta_0)\in\mathbb{R}^d\times\mathbb{R}} L(\mathbf{\theta},\theta_0|\mathcal{D})&\equiv\arg\min_{(\mathbf{\theta},\theta_0)\in\mathbb{R}^d\times\mathbb{R}}\sum^n_{t=1}-\log\Pr (y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\\
\text{Equivalent to minimizing:}\\
\sum^n_{t=1}-\log g\left(y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)&=\sum^n_{t=1}\log\left[1+\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}\right]
\end{aligned}$$

## SGD Implementation

Here, we adopt `SGD` method ([[Gradient Descent#View of dataset]]) and calculate the gradients:

$$\begin{aligned}
\frac{d}{d\theta_0}\log\left[1+\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}\right]&=-y_t\frac{\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}}{1+\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}}\\&=-y_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]\\
\frac{d}{d\mathbf{\theta}}\log\left[1+\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}\right]&=-y_t\mathbf{x}_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]
\end{aligned}$$

With parameter update:
$$\begin{aligned}
\mathbf{\theta}&\gets\mathbf{\theta}+\eta\cdot y_t\mathbf{x}_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]\\
\theta_0&\gets\theta_0+\eta\cdot y_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]
\end{aligned}$$

While setting the gradient to zero is a necessary condition for the optimality of $(\mathbf{\theta}^*,\theta_0^*)$:

$$\begin{aligned}
\frac{d}{d\theta_0}\ell(\mathbf{\theta},\theta_0|\mathcal{D})&=\sum^n_{t=1}-y_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]=0\\
\frac{d}{d\mathbf{\theta}}\ell(\mathbf{\theta},\theta_0|\mathcal{D})&=\sum^{n}_{t=1}-y_t\mathbf{x}_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]=\mathbf{0}
\end{aligned}$$

This triggers several inspirations:
- SGD leads to $\textcolor{red}{\text{no}}$ significant change on average when the gradient of the full objective equals to zero ($\mathbb{E}\Bigl[\nabla\ell(\mathbf{\theta}^*)\Bigr]=\mathbf{0},\mathbb{E}\Bigl[\nabla\ell(\theta_0^*)\Bigr]=0$).
- $\frac{d}{d\theta_0}\ell(\mathbf{\theta},\theta_0|\mathcal{D})$ further shows that $$\sum_{t:y_t=1}\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]=\sum_{t:y_t=-1}\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]$$
	indicating that the false negative rate is equal to the false positive rate, namely the error rate of identifying two classes are the same, indicating the optimality of $\theta_0^*$ is revealed through balanced mistakes even though sample size for each group is not equivalent.


## `Better Understanding on the Optimal Parameters`

If conduct a transform of mapping $\pm1$ labels to $\{0,1\}$ with $\tilde{y}_t=\frac{1+y_t}{2}$:
$$\begin{aligned}
\frac{d}{d\theta_0}\ell(\mathbf{\theta},\theta_0|\mathcal{D})&=\sum^n_{t=1}-y_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]=\sum^n_{t=1}\left[\tilde{y}_t-\Pr(y=1|\mathbf{x}_t,\mathbf{\theta}^*,\theta^*_0)\right]=0\\
\frac{d}{d\mathbf{\theta}}\ell(\mathbf{\theta},\theta_0|\mathcal{D})&=\sum^{n}_{t=1}-y_t\mathbf{x}_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]=\sum^n_{t=1}\mathbf{x}_t\left[\tilde{y}_t-\Pr(y=1|\mathbf{x}_t,\mathbf{\theta}^*,\theta^*_0)\right]=\mathbf{0}
\end{aligned}$$

Intuitively, this turns mistake probabilities into prediction errors, and this indicates that **optimal prediction errors are orthogonal to any linear function of the inputs, i.e., for any $(\mathbf{\tilde{\theta}},\tilde{\theta}_0)$.** This shows no further linearly available information in the training samples to improve the prediction probabilities or mistake probabilities.

In brief, this proof finishes by concluding that

$$\begin{aligned}
\tilde{\theta}_0\sum^n_{t=1}\left[\tilde{y}_t-\Pr(y=1|\mathbf{x}_t,\mathbf{\theta}^*,\theta_0^*)\right]+\left\langle\mathbf{\tilde{\theta}},\sum^n_{t=1}\mathbf{x}_t\left[\tilde{y}_t-\Pr(y=1|\mathbf{x}_t,\mathbf{\theta}^*,\theta_0^*)\right]\right\rangle&=0\\
\text{or:}\ \ \ \sum^n_{t=1}\left(\tilde{\theta}_0+\left\langle\mathbf{\tilde{\theta}},\mathbf{x}_t\right\rangle\right)\left[\tilde{y}_t-\Pr(y=1|\mathbf{x}_t,\mathbf{\theta}^*,\theta_0^*)\right]&=0
\end{aligned}$$

Where the below one is actually a linear function of the inputs. To be more detailed, we have

$$\begin{aligned}
0&=\begin{pmatrix}\tilde{\theta}_0\\\tilde{\theta}_0\\...\\\tilde{\theta}_0\end{pmatrix}^T\cdot\begin{pmatrix}q_1\\ q_2\\ ...\\ q_n\end{pmatrix}+\left\langle\mathbf{\tilde{\theta}},\begin{bmatrix}\mathbf{x}_1&\mathbf{x}_2&...&\mathbf{x}_n\end{bmatrix}\cdot\begin{pmatrix}q_1\\ q_2\\ ...\\ q_n\end{pmatrix}\right\rangle\\
&=\begin{pmatrix}\tilde{\theta}_0\\\tilde{\theta}_0\\...\\\tilde{\theta}_0\end{pmatrix}^T\cdot\begin{pmatrix}q_1\\ q_2\\ ...\\ q_n\end{pmatrix}+\mathbf{\tilde{\theta}}^T\cdot\begin{bmatrix}\mathbf{x}_1&\mathbf{x}_2&...&\mathbf{x}_n\end{bmatrix}\cdot\begin{pmatrix}q_1\\ q_2\\ ...\\ q_n\end{pmatrix}\\
&=\left[\begin{pmatrix}\tilde{\theta}_0\\\tilde{\theta}_0\\...\\\tilde{\theta}_0\end{pmatrix}^T+\mathbf{\tilde{\theta}}^T\cdot\begin{bmatrix}\mathbf{x}_1&\mathbf{x}_2&...&\mathbf{x}_n\end{bmatrix}\right]\cdot\begin{pmatrix}q_1\\ q_2\\ ...\\ q_n\end{pmatrix}\\
&=\Bigl[\tilde{\theta}_0+\left\langle\mathbf{\tilde{\theta}},\mathbf{x}_1\right\rangle\ \ \ \ \tilde{\theta}_0+\left\langle\mathbf{\tilde{\theta}},\mathbf{x}_2\right\rangle\ \ \ \ ...\ \ \ \ \tilde{\theta}_0+\left\langle\mathbf{\tilde{\theta}},\mathbf{x}_n\right\rangle\Bigr]\cdot\begin{pmatrix}q_1\\ q_2\\ ...\\ q_n\end{pmatrix}
\end{aligned}$$

Where $q(t)=\tilde{y}_t-\Pr(y=1|\mathbf{x}_t,\mathbf{\theta}^*,\theta^*_0)$. Therefore, we conclude a further theorem on:

$$\begin{aligned}
\mathbf{z}:=\Bigl(\left\langle\mathbf{\bar{\theta}},\mathbf{x}_t\right\rangle\Bigr)^n_{t=1}, \left\langle\mathbf{z},\mathbf{q}\right\rangle=0
\end{aligned}$$

This is a sufficient and necessary condition for the optimality of the solution, where sufficiency is indicated by the minimal $\ell_2$ norm of $\mathbf{q}$ over all $\mathbf{\theta},\theta_0$. Proof:

$$\begin{aligned}
\text{If $\left\langle\mathbf{z},\mathbf{q}\right\rangle\neq0$, let } \mathbf{q}'&:=\mathbf{q}-\frac{\left\langle\mathbf{z},\mathbf{q}\right\rangle}{\left\langle\mathbf{z},\mathbf{z}\right\rangle}\mathbf{z}\text{, then}\\
\left\langle\mathbf{q}',\mathbf{q}'\right\rangle&=\left\langle\mathbf{q},\mathbf{q}\right\rangle-2\left\langle\mathbf{e},\mathbf{q}\right\rangle+\left\langle\mathbf{e},\mathbf{e}\right\rangle\\
&=\left\langle\mathbf{q},\mathbf{q}\right\rangle-\frac{\left\langle\mathbf{z},\mathbf{q}\right\rangle}{\left\langle\mathbf{z},\mathbf{z}\right\rangle}\left[2\cdot\left\langle\mathbf{z},\mathbf{q}\right\rangle-\left\langle\mathbf{z},\mathbf{q}\right\rangle\right]\leq\left\langle\mathbf{q},\mathbf{q}\right\rangle
\end{aligned}$$

You might make an analogy with the ordinary least squares, where how the final result is related to some spatial  interpretations. The above things help us to understand that when an optimality is reached, the measure of mistake probability values are somehow minimized in the overall sense, thus reaching the maximum of linear models.

# Regularization

Review the `SGD` steps:

$$\begin{aligned}
\mathbf{\theta}&\gets\mathbf{\theta}+\eta y_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]\\
\theta_0&\gets\theta_0+\eta y_t\mathbf{x}_t\left[1-\Pr(y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)\right]
\end{aligned}$$

And likelihood function

$$
L(\mathbf{\theta},\theta_0|\mathcal{D})=\prod^n_{t=1}L\left((\mathbf{\theta},\theta_0)|\mathbf{x}_t,y_t\right)=\prod^n_{t=1}\Pr (y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)=\prod^n_{t=1}\frac{1}{1+\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}}
$$

Infinite scaling of the parameters $\mathbf{\theta},\theta_0$ may lead to a perfect linear classifier would attain the highest likelihood. That is, we can tune $\Pr (y_t|\mathbf{x}_t,\mathbf{\theta},\theta_0)=g\left(y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)=\frac{1}{1+\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}}$ to almost 0 or 1, which successfully separates the linearly or affinely-separable training data.

Therefore, we add a `Regularization` term like $\ell_2$-norm,

$$
\min_{(\mathbf{\theta},\theta_0)\in\mathbb{R}^d\times\mathbb{R}}
\frac{\lambda}{2}\|\mathbf{\theta}\|^2+\sum^n_{t=1}\log\left[1+\exp{\left(-y_t\left(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0\right)\right)}\right]
$$

This also helps the potentially biased estimator to become somehow unbiased.

# Multi-class Regression

Here, instead of logistic regression, we might simply apply the $soft$-$max$ function

$$
\Pr(y=c|\mathbf{x},\{\mathbf{\theta}_c,\theta_{0,c}\}^M_{c=1})=\frac{\exp\left(\left\langle\mathbf{\theta}_c,\mathbf{x}\right\rangle+\theta_{0,c}\right)}{\sum^M_{c'=1}\exp\left(\left\langle\mathbf{\theta}_{c'},\mathbf{x}\right\rangle+\theta_{0,c'}\right)}
$$

Please notice that the logistic regression is the $M=2$ case of the model here.






# Q&A: Why sigmoid function that changes the linear model into [0,1] could represent probability?

The concept of `Generalized Linear Model`:
$$f(y_i)=\exp\left(\frac{y_i\theta_i-b(\theta_i)}{a(\phi)}+c(y_i,\phi)\right)$$
With:
- $y_i$: observed data
- $\theta_i$: linear estimator, i.e., $\theta_i=\mathbf{x}_i^T\mathbf{\beta}$, with $\mathbf{x}_i$ as the feature vector, and $\mathbf{\beta}$ as parameter vector.
- $b(\theta_i)$: cumulant function, a 'connector' between $\theta_i$ and $\mu$.
- $a(\phi)$: dispersion parameter, determining the sparsity of the distribution.
- $c(y_i,\phi)$: normalization term, guaranteeing the validation of the probability distribution.

This is a classical conclusion combined with different distributions like Binomial, Poisson, Exponential distribution, etc. It suggests great properties like:
- $\mu=\mathbb{E}\left[y_i\right]=b'(\theta_i)$. Proof:
	$$\begin{aligned}
	\int f_Y(y; \theta, \phi)  dy &= 1\\
	\int \frac{\partial}{\partial \theta} f_Y(y; \theta, \phi)  dy &= 0\\
	\frac{\partial f_Y}{\partial \theta} = f_Y \cdot \frac{\partial}{\partial \theta} \left( \frac{y\theta - b(\theta)}{a(\phi)} \right) &= f_Y \cdot \frac{y - b'(\theta)}{a(\phi)}\\
	\int f_Y \cdot \frac{y - b'(\theta)}{a(\phi)}  dy &= 0\\
	\frac{\mu - b'(\theta)}{a(\phi)} = 0,&\ \mu=b'(\theta)
	\end{aligned}$$
- $\operatorname{Var} (y_i)=a(\phi)\cdot b''(\theta_i)$. Proof:
  $$\begin{aligned}
	\int \frac{\partial^2}{\partial \theta^2} f_Y(y; \theta, \phi)  dy &= 0\\
	\frac{\partial f_Y}{\partial \theta} &= f_Y \cdot \frac{y - b'(\theta)}{a(\phi)}\\
	\int f_Y \cdot \frac{y - b'(\theta)}{a(\phi)}  dy &= 0\\
	\frac{\partial^2 f_Y}{\partial \theta^2} = \frac{\partial}{\partial \theta} \left( f_Y \cdot \frac{y - b'(\theta)}{a(\phi)} \right) &= \frac{\partial f_Y}{\partial \theta} \cdot \frac{y - b'(\theta)}{a(\phi)} + f_Y \cdot \left( -\frac{b''(\theta)}{a(\phi)} \right)\\
	\frac{\partial^2 f_Y}{\partial \theta^2} &= f_Y \cdot \left( \frac{y - b'(\theta)}{a(\phi)} \right)^2 - f_Y \cdot \frac{b''(\theta)}{a(\phi)}\\
	\int \left[ f_Y \cdot \frac{(y - \mu)^2}{a^2(\phi)} - f_Y \cdot \frac{b''(\theta)}{a(\phi)} \right] dy &= 0\\
	\frac{\text{Var}(Y)}{a^2(\phi)} - \frac{b''(\theta)}{a(\phi)} &= 0
   \end{aligned}$$

If we say $y_i\sim\operatorname{Bernoulli}(\pi_i)$, where $\pi_i$ means when $y_i=1$. Then, the probability density function could be written as:
$$
f(y_i;\pi_i)=\pi_i^{y_i}(1-\pi_i)^{1-y_i}=\exp\left(y_i\log\frac{\pi_i}{1-\pi_i}+\log(1-\pi_i)\right)
$$

Compared with GLM, we get $b(\theta_i)=\log(1-\pi_i),\theta_i=\log\left(\frac{\pi_i}{1-\pi_i}\right)$, we have $\mu_i=\frac{\exp{\theta_i}}{1+\exp{\theta_i}}=\pi_i$, and the canonical link function is calculated as:
$$
g(t)=\left((b')^{-1}\right)(t)=\log\frac{t}{1-t}
$$
Which leads to the conclusion in [[#Formula]].