`Definition`: Choosing a subset of relevant features from the dataset. This is important to improve accuracy, reduce overfitting, and speed up training (especially for ensemble methods).

We have several methods to realize the balance of `Noise, redundancy, slow models (too many features)` and `Missing important patterns (too few features)`.

# Filter

`Definition: Rank features and abandon a subset of features before modeling using statistical methods.`

We will introduce a few methods to do this.

## Method: Variance

Dataset $D={(x_{t}, y_{t})}_{t=1}^{N}$ , each sample $x_{t}=(x_{t, 1}, ..., x_{t, d})$, we conduct **Variance of numerical feature**: 
$$\overline{x_{(i)}}=\frac{1}{N} \sum_{t=1}^{N} x_{t, i}, \text{Var}^{(i)}=\frac{1}{N} \sum_{t=1}^{N}\left(x_{t, i}-\overline{x_{(i)}}\right)^{2}$$
And we keep feature $\{x_{t, i}\}_{t=1}^{N}$ with high variance.

## Method: Relevance between Feature and Target

Pearson Correlation between continuous variables $$r^{(i)}=\frac{\sum_{i=t}^{n}\left(x_{t, i}-\overline{x_{(i)}}\right)\left(y_{t}-\overline{y}\right)}{\sqrt{\sum_{i=t}^{n}\left(x_{t, i}-\overline{x_{(i)}}\right)^{2}} \sqrt{\sum_{t=1}^{n}\left(y_{t}-\overline{y}\right)^{2}}}$$
And we keep feature $\{x_{t, i}\}_{t=1}^{N}$ with high correlation.

## Method: Tested Correlation

Chi-squared between categorical variables: 
- $\{x_{t, i}\}_{t=1}^{N}$ take values in $\{a_{1}, a_{2}, ...\}$ $\{y_{t}\}_{t=1}^{N}$ take values in $\{b_{1}, b_{2}, ...\}$ $$P_{j}=\left|\left\{t: x_{t, i}=a_{j}\right\}\right|, Q_{k}=\left|\left\{t: y_{t}=b_{k}\right\}\right|$$ $$O_{j k}=\left|\left\{t: x_{t, i}=a_{j}, y_{t}=b_{k}\right\}\right|$$ $$\chi_{(i)}^{2}=\sum_{j, k} \frac{\left(O_{j k}-E_{j k}\right)^{2}}{E_{j k}}, E_{j k}=\frac{P_{j} \cdot Q_{k}}{n} .$$
- Higher $\chi_{(i)}^2\Rightarrow$ Smaller p-value $(<0.05 ? ) \Rightarrow Correlated$!
- Keep feature $\{x_{t, i}\}_{t=1}^{N}$ with small p -value

## Method: Mutual Information

The mutual information formula is given by
$$\begin{aligned}
\text{Discrete: }I(X;Y)&=\sum p(x)p(y)\log\frac{p(x,y)}{p(x)p(y)}\\
\text{Continuous: }I(X;Y)&=\int_Y\int_X f_X(x)f_Y(y)\log\frac{f_{X,Y}(x,y)}{f_X(x)f_Y(y)}dxdy
\end{aligned}$$

A higher mutual information value indicate higher correlations, which should be kept.

# Wrapper

`Definition`: test features by training a model. Two basic ideas are introduced.

`Method 1: Sequential Forward Selection`
With initialization of none, every iteration we select a feature that maximizes the reduction in error.

`Method 2: Recursive Feature Elimination`
With initialization of all features, every iteration we select a feature that minimizes the gain in error.

`Pros & Cons`:
1. Pros: Model specific, Considering interactions between features
2. Cons: Slow, can overfit to the training data

# Embedded Methods

Selection during model training. This could be done by:
- Use regularizer to penalize unimportant features, e.g., Lasso, Ridge.
- Decision trees naturally perform feature selection during training.


There are also pros and cons. For example:
1. Pros
	- Efficient
	- Tailored to Models
2. Cons
	- Depends on the model choice