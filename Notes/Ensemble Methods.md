**Table of Contents**
[[#Intro Decision Tree]]
[[#Ensemble Learning Preview]]
[[#Ensemble Learning BAGGING]]


# Intro: Decision Tree

Please refer to [[Decision Tree]] if needed. Basically:

**Decision tree learning** is a [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning "Supervised learning") approach used in [statistics](https://en.wikipedia.org/wiki/Statistics "Statistics"), [data mining](https://en.wikipedia.org/wiki/Data_mining "Data mining") and [machine learning](https://en.wikipedia.org/wiki/Machine_learning "Machine learning"). In this formalism, a classification or regression [decision tree](https://en.wikipedia.org/wiki/Decision_tree "Decision tree") is used as a [predictive model](https://en.wikipedia.org/wiki/Predictive_model "Predictive model") to draw conclusions about a set of observations.
A tree is built by splitting the source [set](https://en.wikipedia.org/wiki/Set_\(mathematics\) "Set (mathematics)"), constituting the root node of the tree, into subsets—which constitute the successor children. The splitting is based on a set of splitting rules based on classification features. This process is repeated on each derived subset in a recursive manner called [recursive partitioning](https://en.wikipedia.org/wiki/Recursive_partitioning "Recursive partitioning"). The [recursion](https://en.wikipedia.org/wiki/Recursion "Recursion") is completed when the subset at a node has all the same values of the target variable, or when splitting no longer adds value to the predictions. This process of _top-down induction of decision trees_ (TDIDT) is an example of a [greedy algorithm](https://en.wikipedia.org/wiki/Greedy_algorithm "Greedy algorithm"), and it is by far the most common strategy for learning decision trees from data.

Basically, as a base learner, decision tree model might not perform well though compatible with different tasks. This leads us to think about ways that could empower such model to perform better other than tuning the hyperparameters. One way is to utilize `the wisdom of crowds`.

# Ensemble Learning: Preview

`Informal Definition`: Methods to improve the performance of weak learners. In this way, the prediction responsibility is shifted from 1 weak learner to an "ensemble" of several weak learners. A simplest way is to get a set of weak learners to form a strong learner by `majority vote`.

For example, when conducting `binary classification`, we could realize majority vote by figuring out which class is mostly supported. For $x_i$ as prediction task, $h_i$ as learner, and $f$ as the mode of the result for task $x_i$ for all learners, there are several cases to be determined:

|                         |           $x_1$           |           $x_2$           |           $x_3$           |         $x_4$          |           $x_5$           |           $x_6$           |           $x_7$           |           $x_8$           |
| :---------------------: | :-----------------------: | :-----------------------: | :-----------------------: | :--------------------: | :-----------------------: | :-----------------------: | :-----------------------: | :-----------------------: |
|          $h_1$          |  $\textcolor{green}{✓}$   |  $\textcolor{green}{✓}$   | $\textcolor{red}{\times}$ | $\textcolor{green}{✓}$ | $\textcolor{red}{\times}$ |  $\textcolor{green}{✓}$   | $\textcolor{red}{\times}$ | $\textcolor{red}{\times}$ |
|          $h_2$          |  $\textcolor{green}{✓}$   | $\textcolor{red}{\times}$ |  $\textcolor{green}{✓}$   | $\textcolor{green}{✓}$ | $\textcolor{red}{\times}$ | $\textcolor{red}{\times}$ |  $\textcolor{green}{✓}$   | $\textcolor{red}{\times}$ |
|          $h_3$          | $\textcolor{red}{\times}$ |  $\textcolor{green}{✓}$   |  $\textcolor{green}{✓}$   | $\textcolor{green}{✓}$ | $\textcolor{red}{\times}$ | $\textcolor{red}{\times}$ | $\textcolor{red}{\times}$ |  $\textcolor{green}{✓}$   |
|           $f$           |  $\textcolor{green}{✓}$   |  $\textcolor{green}{✓}$   |  $\textcolor{green}{✓}$   | $\textcolor{green}{✓}$ | $\textcolor{red}{\times}$ | $\textcolor{red}{\times}$ | $\textcolor{red}{\times}$ | $\textcolor{red}{\times}$ |
| Effect of Majority Vote |             +             |             +             |             +             |           0            |             0             |             -             |             -             |             -             |

Among which, $x_1,x_2,x_3$ shows stronger performance of $f$ even though there are error for single learners, and $x_6,x_7,x_8$ shows weaker performance of $f$ even though there are correct items. This intrigues us to care about the error rate $\epsilon$ of each base learner.

`Effectiveness of Majority Vote`:

Suggests that when conducting `binary classification`, we have all the learners $X_1,X_2,...,X_N\sim \operatorname{Bern}(1-\epsilon)$ are *`i.i.ds`*. Therefore, we have:

$$\begin{aligned}
\Pr(\hat{f}\text{ fails})&=\sum^{\lfloor N/2\rfloor}_{k=0}\begin{pmatrix}N\\ k\end{pmatrix}(1-\epsilon)^k\epsilon^{N-k}\\
&\leq\Pr\left(N\cdot\overline{X}_N\leq\frac{N}{2}\right)\\
&=\Pr\left(N\cdot\overline{X}_N-\mathbb{E}\left[\overline{X}_N\right]\leq\frac{N}{2}-\mathbb{E}\left[\overline{X}_N\right]\right)\\
&=\Pr\left(N\cdot\overline{X}_N-N\cdot(1-\epsilon)\leq\frac{N}{2}-N\cdot(1-\epsilon)\right)\\
&=\Pr\left(\overline{X}_N-(1-\epsilon)\leq-(\frac{1}{2}-\epsilon)\right)\\
&\leq\exp\left(-\frac{1}{2}N(1-2\epsilon)^2\right)
\end{aligned}$$

Which is conducted by `Hoeffding's Inequality`:
==Let $X_1,X_2,...,X_n$ be independent bounded random variables with $X_i\in[a,b],\forall i$, where $-\infty<a\leq b<\infty$. Then for $\delta>0$ we have==
$$\begin{aligned}
P(\frac{1}{n}\sum^n_{i=1}X_i-\frac{1}{n}\mathbb{E}[X_i]\leq-\delta)&\leq\exp\left(-\frac{2N\delta^2}{(b-a)^2}\right)\\
P(\frac{1}{n}\sum^n_{i=1}X_i-\frac{1}{n}\mathbb{E}[X_i]\geq\delta)&\leq\exp\left(-\frac{2N\delta^2}{(b-a)^2}\right)
\end{aligned}$$

Where $\delta=\frac{1}{2}-\epsilon$ if $\epsilon<\frac{1}{2}$.

This indicates that when the error is lower than random guess, then with the increase of base learners, the error rate drops exponentially. On the other hand, when the error is greater than random guess, then with the increase of base learners, the rate of correctness drops exponentially.

This intrigues us to seek for more diverse (since the hypothesis of above are ideal about independence) and accurate base learners (with adoptable error rate).


# Ensemble Learning: BAGGING

`BAGGING (Bootstrap AGGregatING)` is a way to create diverse base learners based on different training samples each base learner receives. This differentiate leaners with different subsets of the training dataset. And for BAGGING itself, it combines the model outputs to lower the variance of each base learner since random oscillations exist. This is conducted by `majority vote (classification)` and `average (regression)`.

In BAGGING, we apply `Bootstrap Sampling` on the dataset, which is randomly selecting samples with replacement. This allows same samples to be sampled multiple times. Here's how bagging works:
- Take the original dataset
- Generate $N$ bootstrap samples
- Each set is the same size as the original but slightly different (by sampling with replacement)

Then, for `training`, we have:
- Train one model on each bootstrap sample
- Each model learns slightly different patterns due to sample variation
- *Then, we get $N$ weak learners*

After training, we conduct `aggregation` by:
- For classification: Use majority voting (e.g., 6 trees say ”yes,” 4 say ”no” ￫”yes”).
- For regression: Average the predictions.
- **These are conducted to reduce individual errors.**

Therefore, since a sample could be selected multiple times, the method make more use of single data point compared to partitioning the data into $N$ subsets. However, it is worth to notice that some data are not selected at all (Out-of-Bag Samples), reaching an expected minimum of
$$\begin{aligned}
\mathbb{E}\left(\text{ratio of OOB}\right)&=\frac{1}{n}\mathbb{E}\left(\sum^n_{i=1}\mathbb{1}\{x_i\text{ is OOB}\}\right)\\
&=\frac{1}{n}\sum^n_{i=1}\mathbb{E}(\mathbb{1}\{x_i\text{ is OOB}\})\\
&=\frac{1}{n}\sum^n_{i=1}\Pr(x_i\text{ is OOB})\\
&=\frac{1}{n}\sum^n_{i=1}(1-\frac{1}{n})^n\\
&=(1-\frac{1}{n})^n\geq\frac{1}{e}\approx0.368
\end{aligned}$$

Such OOB samples could be utilized so as to:
- Cross Validation by calculating the errors in validation sets.
- Hyperparameter Tuning
- Early Stopping (when increasing the number of base learners)

These are somehow conducted by:
1. Find all models that are not trained by a OOB data sample $x_{i_{0}}$
2. Take the majority vote of these models’ result for this OOB sample $x_{i_{0}}$ , compared to the true value of the OOB sample $x_{i_{0}}$
3. Compile the OOB error for all OOB samples in the OOB dataset

This defines a way (**an error metric**) to evaluate ensemble methods, which is calculated via cross-validation without additional methods to split the dataset.

There are several benefits for BAGGING:
- Easy to Implement
- Linear Training/Inference Cost
- Easy to Parallelize
- Model Agnostic

## Error Exploration

Also, BAGGING helps `variance reduction` in prediction. Basically, referring to [[Model Evaluation and Choice#^9a1bd6|No Free Lunch]], we might figure out the error by `bias-variance decomposition`:
$$\begin{aligned}
\mathbb{E}\left[(g-\hat{f})^2\right]&=\mathbb{E}\left[(\hat{f}-\mathbb{E}(\hat{f})+\mathbb{E}(\hat{f})-g)^2\right]\\
&=\mathbb{E}\left[(\hat{f}-\mathbb{E}(\hat{f}))^2\right]+\mathbb{E}\left[(\mathbb{E}(\hat{f})-g)^2\right]+2\cdot\mathbb{E}\left[(\hat{f}-\mathbb{E}(\hat{f}))(\mathbb{E}(\hat{f})-g)\right]\\
&=\mathbb{E}\left[(\hat{f}-\mathbb{E}(\hat{f}))^2\right]+\mathbb{E}\left[(\mathbb{E}(\hat{f})-g)^2\right]+2(\mathbb{E}(\hat{f})-g)\cdot\mathbb{E}\left[(\hat{f}-\mathbb{E}(\hat{f}))\right]\\
&=\mathbb{E}\left[(\hat{f}-\mathbb{E}(\hat{f}))^2\right]+\mathbb{E}\left[(\mathbb{E}(\hat{f})-g)^2\right]+0\\
&=\text{Variance}+\text{Bias}^2
\end{aligned}$$

Moreover, the actual situation includes inevitable random noise, which is:
$$\mathbb{E}[(\hat{f}+\varepsilon-g))^{2}]=\underbrace{(g-\mathbb{E} \hat{f})^{2}}_{bias }+\underbrace{\mathbb{E}\left[(\hat{f}-\mathbb{E} \hat{f})^{2}\right]}_{variance }+\underbrace{Var(\varepsilon)}_{Noise}$$

However, the ensemble methods effectively lowers variance when combining the output of multiple base learners, which reduces the variance and further reduces the expected error. Although it seems to contradict with `no free lunch` theorem, the hypothesis is actually changed, which doesn't match the assumption of the theorem.

**When is the variance reduced?**

For $i.i.d$ samples we might suggest that:
$$Var(\overline{X})=Var\left(\frac{1}{B} \sum_{i=1}^{B} X_{i}\right)=\frac{1}{B^{2}} Var\left(\sum_{i=1}^{B} X_{i}\right)=\frac{1}{B^{2}} B \sigma^{2}$$

Which effectively shrinks the variance by $\frac{1}{B^2}$. However, for only $i.d$ samples (identically distributed), we have:
$$Var\left(\sum_{i=1}^{B} X_{i}\right)=\sum_{i=1}^{B} Var\left(X_{i}\right)+\sum_{i \neq j} Cov\left(X_{i}, X_{j}\right)=B \sigma^{2}+B(B-1) \rho \sigma^{2}$$

Where:
$$\begin{aligned}
\rho_{X, Y}&=corr(X, Y)=\frac{cov(X, Y)}{\sigma_{X} \sigma_{Y}}=\frac{E\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]}{\sigma_{X} \sigma_{Y}}, \text{if } \sigma_{X} \sigma_{Y}>0\\
\rho_{X, Y}&=\frac{E(X Y)-E(X) E(Y)}{\sqrt{E\left(X^{2}\right)-E(X)^{2}} \cdot \sqrt{E\left(Y^{2}\right)-E(Y)^{2}}}
\end{aligned}$$

And worse still, for correlated learners with $\rho\to1$, we have the variance $\to\sigma^2$, indicate no progress in variance reduction. Therefore, we would hope the base learners to be as independent as possible (thus uncorrelated). Sadly, the subsets sampled are somehow correlated, since they contain part of identical samples.

## Codes

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
def train_model(X_train, y_train):
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    return tree
def bootstrap(X_train, y_train, n_bootstrap_sample, n_estimators, return_indices=False):
    bootstrap_samples = []
    bootstrap_indices = []
    for _ in range(n_estimators):# Generate bootstrap sample, using the
    # np.random.choice function, setting the parameter `replace` as true.
        indices = np.random.choice(X_train.shape[0], n_bootstrap_sample,
									replace=True)
        X_bootstrap = X_train[indices]
        y_bootstrap = y_train[indices]
        bootstrap_samples.append((X_bootstrap, y_bootstrap))
        bootstrap_indices.append(indices)
    if return_indices:
        return bootstrap_samples, bootstrap_indices
    return bootstrap_samples

def majority_voting(predictions):
    return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(),
							    axis=0, arr=predictions)

def bagging_predict(model_list, X):
    predictions = np.array([model.predict(X) for model in model_list])
    return majority_voting(predictions)

bootstrap_samples = bootstrap(X_train, y_train, X_train.shape[0], n_estimators=11)
model_list = [train_model(X_train, y_train) for X_train, y_train in
				bootstrap_samples]
bagged_prediction = bagging_predict(model_list, X_test)
```

# Ensemble Methods: Random Forest

Consider decision trees in BAGGING. Certainly, any operations on single base learners, like `Pruning`, are not preferred since:
- Decision Trees are unstable classifiers, with different subsets potentially yielding significantly different trees.
- We want them to be in low bias but high-variance when conducting BAGGING.
- Unpruned/Unoperated trees tend to differ more from one another.

Then we introduce `Random Forest` to try further decorrelating the trees. This algorithm conducts decorrelation in two dimensions:
- BAGGING: by using different training subsets
- Limiting Tree Complexity: the trees are limited, considering $\textcolor{red}{\text{only k features}}$ from the full feature set $m$. That is:
	- $k<m$, which further limits on the tree depth.
	- $k=m$, which degrades to BAGGING method.

Limiting the complexity of trees is conducive when doing ensembles, since the robustness of tree is further guaranteed, with overlapping studied features between trees are reduced. Also, certain features of BAGGING are guaranteed, e.g., OOB samples' leading to natural cross-validation.

## Tuning Random Forests

Random forest models have multiple hyper-parameters to tune. For example,
- Level of each base learner:
	- number of features to randomly select at each split
	- minimum number of samples required to be at a leaf node
	- ...
- Level of the whole ensemble:
	- total number of trees in the ensemble (min_samples_leaf)

## Pros & Cons

Pros:
- Works great in practice, with $k$ to be treated as a hyperparameter
- Introduce diversity through
	- randomness in bootstrapping
	- randomness in feature selection
- Faster training than bagging, with less features in base trees.

Cons:
- Less likely to select meaningful features since they are selected randomly.
- Biased predictions for imbalanced datasets (not evenly distributed), since the majority learn badly.

## Codes

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]])) # output:[1]
```

# Ensemble Methods: Boosting

This introduces boosting thoughts that addresses the shortcomings of single decision tree models in some other way: `learn from mistakes (errors)`.

The core idea of `Boosting Models` is to take an ensemble of simple models $\{T_h\}h\in\mathcal{H}$ and additively combine them into a single, more complex model. Each model $T_{h}$ might be a poor fit for the data, but a linear combination of the poor ensembles can be expressive:
$$T=\sum_h\lambda_h T_h$$

Then the task goes to how $\lambda_h$ is determined and how $T_h$ is trained.

## Gradient Boosting

The core idea is to **build new models to compensate the error of the current ensemble**. Therefore, this algorithm adopt key features like:
- A `more general boosting method` using gradient descent
- Works with `any differentiable loss function`
- Each new model fits the `'residual errors'` of the previous ones

### Procedure

1. Fit a simple model $T=T^{(0)}$ on the training data ${(x_{1}, y_{1}), ...,(x_{N}, y_{N})}$
2. Compute the residuals (‘errors’) ${r_{1}, ..., r_{N}}$ for model T
3. Fit a simple model $T^{(1)}$ to the current residuals, i.e., train using ${(x_{1}, r_{1}), ...,(x_{N}, r_{N})}$
4. Build a new model $T \leftarrow T+\lambda T^{(1)}$
5. Compute the updated residuals $r_{n} \leftarrow r_{n}-\lambda T^{(1)}(x_{n})$
6. Repeat steps 2-5 until the stopping condition met

### Understanding the Loss Function

Recap [[Gradient Descent#Methodology]] and consider the `MSE` loss, which is:
$$\mathcal{L}_{\text{MSE}}=\sum^n_{i=1}(y_i-\hat{y}_i)^2$$

Through gradient, we have $\nabla\mathcal{L}_{\text{MSE}}=-2[y_1-\hat{y}_1,y_2-\hat{y}_2,...y_n-\hat{y}_n]^T=-2[r_1,...,r_n]^T$. Therefore, we update the variable the variable $\hat{y}_i\gets\hat{y}_i+\lambda r_i$. From this point of view, $T^{(k)}$ is more like `an estimator on the gradient`, which is further added to the ensemble model with learning rate $\lambda$.

Like gradient descent, we have shrinkage parameter (learning rate) $\lambda$ to tune (along with number of trees $B$, number of splits in each tree $d$, etc.), and variants like:
- Tree constraints
- Weighting each tree to the additive sum using a learning rate
- Sampling strategies: stochastic gradient boosting
- Regularization: $\ell_1$-norm/$\ell_2$-norm
are introduced `at the level of ensemble model` (not just tree models). Successful trials are XGBoost, LightGBM, etc.

### Codes

Below is a demo of plotting gradient boosting (linear base models).
```python
def compute_residual(y, y_pred):
    """
    Compute the negative gradient (residual) for gradient boosting.
    Returns derivative of L2 loss w.r.t f
    """
    return y - y_pred
def train_gradient_boosting(
    X,
    y,
    n_estimators, # number of iterations
    learning_rate, # alpha
    axs):
    """
    Train a gradient boosting model with Linear base learners.
    """
    # Initialize with the mean of y
    y_pred = np.mean(y)
    axs[0].plot(X, y_pred * np.ones(50), label=f'Gradient Boosting (i={0})',
			    linestyle='--', color='blue', alpha = 1 / n_estimators)
    # Iterate to fit n_estimators base learners
    for _ in range(n_estimators):
        # Compute the negative gradient (residual)
        residuals = compute_residual(y, y_pred) 
        # Fitting linear base learner (This the answer for the first question)
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), residuals)
        # Update the predictions with the spline approximation
        y_pred += learning_rate * model.predict(X.reshape(-1, 1)) # (**this part
											        # will be used as a question**)
        axs[0].plot(X, y_pred, label=f'Gradient Boosting (i={_})', linestyle='--',
			        color='blue', alpha = _ / n_estimators)
        axs[1].plot(X, residuals, label=f'Residuals (i={_})', linestyle='--',
			        color='blue', alpha = _ / n_estimators, marker='x', mfc='red',
			        mec='red')
    return y_pred, residuals
```

## AdaBoost

### Motivation
Gradient Boosting conducts remarkable tasks on regression analysis with `differentiable loss` functions, but worse when conducting classification tasks with loss function like
$$\mathcal{L}=\frac{1}{n}\sum^n_{i=1}\mathbb{1}(y_i\neq \hat{y}_i)$$

Such loss functions are not differentiable, making the estimation on the gradient difficult. One way is to change to a differentiable loss function (`exponential loss`):
$$\mathcal{L}_{\exp }=\frac{1}{n}\sum_{i=1}^{n} \exp \left(-y_{i} \hat{y}_{i}\right), y_{i} \in\{-1,1\}$$

This loss function excels at:
- Differentiability
- Good indicator of classification error ($\text{correct}=1/e,\text{error}=e$)
In this loss function, **classification task is automatically transferred into a regression task**. Bias in task definition, however, brings benefits of taking the gradients:
$$\begin{aligned}\frac{\partial \mathcal{L}_{\exp}}{\partial \hat{y}_i}&=\frac{1}{n}(-y_i)\cdot\exp \left(-y_{i} \hat{y}_{i}\right)\\
\Rightarrow \nabla_{\mathbf{\hat{y}}}\mathcal{L}_{\exp}&=\frac{1}{n}\left[-y_1\exp(-y_1\hat{y}_1,...-y_n\exp(-y_n\hat{y}_n)\right]^T\\
&:=\left[w_1y_1,...,w_ny_n\right]^T
\end{aligned}$$

This introduces `AdaBoost`.

### Procedure (for binary task)

1. Initialize equal weights $1 / n$ for all training samples
2. For each iteration:
	(a) Train a weak learner $T^{(t)}$ on weighted data $\{(x_{1}, w_{1} y_{1}), ...,(x_{N}, w_{N} y_{N})\}$ and get the error rate $\epsilon^{(t)}$
	(b) If $\epsilon^{(t)}>0.5$ then break (**while $<0.5$ might not be alright**)
	(c) Compute the classifier weight $\lambda^{(t)}=\frac{1}{2} \ln (\frac{1-\epsilon^{(t)}}{\epsilon^{(t)}})$
	(d) Update sample weights $w_{i}^{(t+1)} \leftarrow w_{i}^{(t)} \exp (-\lambda^{(t)} y_{i} T^{(t)}(x_{i})) / Z$ where Z is the normalizing constant
	(e) Update the model $T \leftarrow T+\lambda^{(t)} T^{(t)}$
3. Combine all weak classifiers $T=\sum_{t} \lambda^{(t)} T^{(t)}$

Where several equations require further explanations.
1. **2(c)** $\lambda^{(t)}=\frac{1}{2} \ln (\frac{1-\epsilon^{(t)}}{\epsilon^{(t)}})$
	$\lambda^{(t)}$ should minimize exponential loss of the learner $T^{(t)}$$$\begin{aligned}\frac{\partial \mathcal{L}_{\exp }}{\partial \lambda^{(t)}}=\frac{\partial \sum_{i} \exp \left(-y_{i} \lambda^{(t)} \hat{y}_{i}^{(t)}\right)}{\partial \lambda^{(t)}}=-N_{correct } \exp \left(-\lambda^{(t)}\right)+N_{wrong } \exp \left(\lambda^{(t)}\right)=0\\\lambda^{(t)}=\frac{1}{2} ln \left(\frac{N_{correct }}{N_{wrong }}\right)=\frac{1}{2} ln \left(\frac{N_{correct } /\left(N_{correct }+N_{wrong }\right)}{N_{wrong } /\left(N_{correct }+N_{wrong }\right)}\right)=\frac{1}{2} ln \left(\frac{1-\epsilon^{(t)}}{\epsilon^{(t)}}\right)\end{aligned}$$
2. **2(a)** $T^{(t)}$ trained on $\{(x_{1}, w_{1} y_{1}), ...,(x_{N}, w_{N} y_{N})\}$
	Obtain $T^{(t+1)}$ that minimize exponential loss for $T+T^{(t+1)}$ $$\begin{aligned} & \mathbb{E}_{(x, y) \in D}\left[\exp \left(-y\left(T(x)+T^{(t+1)}(x)\right)\right]\right. \\ = & \mathbb{E}_{(x, y) \in D}\left[\exp (-y T(x)) \exp \left(-y T^{(t+1)}(x)\right)\right] \\ \approx & \mathbb{E}_{(x, y) \in D}\left[\exp (-y T(x))\left(1-y T^{(t+1)}(x)+\frac{1}{2} y^{2} T^{(t+1)}(x)^{2}\right)\right] (\textcolor{green}{\text{Taylor expansion}}) \\ = & \mathbb{E}_{(x, y) \in D}\left[\exp (-y T(x))\left(\frac{3}{2}-y T^{(t+1)}(x)\right)\right] (\textcolor{green}{\text{Binary classification}: y, T^{(t+1)}(x)= \pm 1}) \end{aligned}$$
	Since the choice of $T^{(t+1)}$ $\textcolor{red}{\text{CANNOT}}$ change $\mathbb{E}_{(x, y) \in D}[\exp (-y T(x)) \cdot 3 / 2]$, we have
	$$\begin{aligned} T^{(t+1) *} & =\underset{T^{(t+1)}}{\arg \max } \mathbb{E}_{(x, y) \in D}\left[\exp (-y T(x)) y T^{(t+1)}(x)\right] \\ & =\underset{T^{(t+1)}}{\arg \max } \mathbb{E}_{(x, y) \in D}\left[\frac{\exp (-y T(x))}{\mathbb{E}_{(x, y) \in D} \exp (-y T(x))} y T^{(t+1)}(x)\right] \\ & =\underset{T^{(t+1)}}{\arg \max } \mathbb{E}_{(x, y) \in D^{(t+1)}}\left[y T^{(t+1)}(x)\right](\text{by }\textcolor{green}{D^{(t+1)}(x) \propto D(x) \exp \left(-y\left(\sum_{i=1}^{t} \lambda^{(i)} T^{(i)}(x)\right)\right)}) \end{aligned}$$

### Codes

Below is a demo of classification by AdaBoost:
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]]) # output: array([1])
clf.score(X, y) # output:0.96
```

## Pros & Cons

Pros:
1. High Accuracy:
	- AdaBoost: Turns weak learners into strong classifiers (e.g., great for spam detection)
	- Gradient Boosting: Often beats other methods.
2. Flexibility:
	- AdaBoost: Simple and fast for binary tasks.
	- Gradient Boosting: Handles any (differentiable) loss function - regression, classification, ranking.
3. Both can Rank Feature Importance (especially for gradient boosting).

Cons:
1. Overfitting Risk:
	- AdaBoost: Sensitive to noisy data and outliers (overemphasizing them).
	- Gradient Boosting: Too many trees or poor tuning can overfit.
2. Computational Cost:
	- Both: Sequential training - slower than parallel methods like bagging
	- Gradient Boosting: Especially resource-heavy with many iterations.
3. Tuning (e.g., learning rate, rounds, etc.) needed for best results.

## Strategies

1. Simple weak learners (e.g., stumps, 1-layer decision tree) are OK for best results
2. Limit rounds (e.g., 50-100) to avoid overfitting
3. Preprocess data to reduce noise and outliers.




# Credits

Wikipedia contributors. (2025, July 9). _Decision tree learning_. Wikipedia. https://en.wikipedia.org/wiki/Decision_tree_learning.

