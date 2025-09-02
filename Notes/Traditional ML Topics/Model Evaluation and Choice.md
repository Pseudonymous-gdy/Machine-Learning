
**Document Class** #DSAA2011  #Evaluation #Supervised-Learning #introduction 

**Table of Contents:**
- [[#Generalizability of Supervised Learning Algorithms|Generalizability of Supervised Learning Algorithms]]
	- [[#Generalizability of Supervised Learning Algorithms#Definition & Machine Learning Process|Definition & Machine Learning Process]]
- [[#Error Metrics|Error Metrics]]
- [[#Regulizer|Regulizer]]
- [[#Variance and Bias|Variance and Bias]]
- [[#Overfitting and Underfitting|Overfitting and Underfitting]]
- [[#Data Division and Cross Validation|Data Division and Cross Validation]]
- [[#Future Reading|Future Reading]]

**Ways for us to evaluate supervised learning models.**

---
## Generalizability of Supervised Learning Algorithms

The basic motivation is that `we need the new ML system generalizes well on unseen data`. Therefor we develop a series of vocabulary to conduct study on models.

### Definition & Machine Learning Process

`Unknown target function`: $f:\mathcal{X}\to\mathcal{Y}$. This is what we hope to approach.

`Training Examples (dataset)`: $\mathcal{D}:=\{(x_1,y_1),...,(x_n,y_n)\}$

`Hypothesis set`: $\mathcal{H}$. It's a set of functions we assume $f$ to be.

`Learning algorithm`: $\mathcal{A}$. It's the algorithm that machine learns based on hypothesis.

`Final hypothesis`: $g:\mathcal{X}\to\mathcal{Y}$

**Machine Learning Process**: take linear regression for example

1. `Learning/Training`: Given a `dataset`, we conduct:
	- `Hypothesis class`: Design an affine model such that $f_{\bar{\mathbf{w}}}(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b=y$.
	- `Learning algorithm`: Find the minimizer of loss function$$J(\overline{\mathbf{w}})=(\mathbf{X\overline{w}}-\mathbf{y})^T(\mathbf{X\overline{w}}-\mathbf{y})$$
	- `Final hypothesis`: Least squares solution$$\mathbf{\overline{w}}^*=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
2. `Prediction/Testing`: Given a new feature vector, make prediction based on the least squares solution.

## Error Metrics

The main idea is to `Estimate the Performance for New Data`. Therefore, besides training and testing, there should be one part called `Validation Dataset` to fit the hyperparameters, while to determine the hyperparameters requires certain metrics.

`Error Metric`: we define an error function$\operatorname{error}:\mathcal{Y}\times\mathcal{Y}\to\mathbb{R}$ such that
$$
\operatorname{error}(y,y')=\text{ penalty for }\textcolor{red}{\text{predicting }y'}\text{ when }\textcolor{blue}{\text{true label is }y}
$$

Specially, for Regression tasks, the error functions we've studied are:
- Square error: $\operatorname{error}_{sq}(y,y')=(y-y')^2$
- Absolute error: $\operatorname{error}_{abs}(y,y')=|y-y'|$

For Classification tasks, the error functions we could adopt are:
- Misclassification error: $\operatorname{error}_{mis}(y,y')=\mathbb{1}\{y\neq y'\}$
- Weighted misclassification error: $\operatorname{error}_{\beta}(y,y')=\beta\mathbb{1}\{y'=1,y=-1\}+(1-\beta)\mathbb{1}\{y'=-1,y=1\}$, where false positives are $\beta$ times worse than false negatives.
- Balanced error rate:
	$$
	\operatorname{error}_{bal}(y,y')=\begin{cases}\frac{1}{n_+},y'\neq y=1\\\frac{1}{n_-}, y'\neq y=-1\\0,\quad y=y'\end{cases}
	$$
	This might be multiplied by a factor for ranging the error between 0 and 1.
	Also, this is also defined as:
	$$
	\operatorname{BER}(\mathbf{y},\mathbf{y}')=\frac{1}{2}(\frac{mis_+}{n_+}+\frac{mis_-}{n_-})$$

Besides error metrics, we hope to further analyze the performance (or scores) of a model, especially classification model. Therefore we introduce confusion matrix.

`Confusion Matrix`:

|               |                             | Predicted    | Condition    |
| ------------- | --------------------------- | ------------ | ------------ |
|               | **Total Population (=P+N)** | **Positive** | **Negative** |
| **Actual**    | **Positive**                | TP           | FN           |
| **Condition** | **Negative**                | FP           | TN           |
This matrix describes all kinds of errors and correctness of one model, from which we could derive scores to describe the performances.

`Precision`: $Precision=\frac{TP}{TP+FP}$
`Recall (Sensitivity)`: $Recall=\frac{TP}{TP+FN}$
`Accuracy`: $Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$

`F1 Score`: $F1 Score=\frac{1}{Precision^{-1}+Recall^{-1}}$

Also, there are some ways to further evaluate classification model, which regards to ROC curves.

`True Positive Rate`$:=\frac{TP}{TP+FN}$
`False Positive Rate`$:=\frac{FP}{FP+TN}$

`ROC (Receiver Operating Characteristics)`: A plot on True Positive Rate by False Positive Rate

*How to Plot*: After receiving a set of predicted probability or score, the algorithm starts from score=1, and gradually lower the scores to calculate the `TPR` and `FPR`. **This helps to remove the influence of the thresholds that determine the final result of classes.** Often, we adopt the threshold that is closet to $(0,1)$ in the plot since it refers to 100% `TPR` and 0% `FPR`.

`AUC`: The area under ROC curve.

**For Multi-class Models**:
- calculate precision, recall within each class
- average them across all classes (you might consider `balanced averages`)

## Regulizer

`Pros`:
- Reduce the variance of the model to make it more stable
- Prevent overfitting
- Impose prior structural knowledge
- Improve interpretability: simpler model, automatic feature selection

`Cons`:
- Gauss-Markov theorem: least squares is the best linear `unbiased` estimator.
- Regularization increase biases.

`Types of Regulizer`:
We have $\ell_p$-norm, which is
$$
\|\mathbf{w}\|_p=\Bigl(\sum |w_i|^p\Bigr)^\frac{1}{p}
$$
Specially, for $p=0,\infty$, we have
$$\begin{aligned}
\|\mathbf{w}\|_0&=\sum1=\dim(\mathbf{w})\\
\|\mathbf{w}\|_\infty&=\max_{i={1,2,...,\dim{\mathbf{w}}}}w_i
\end{aligned}$$
This is also a type of distance, since according to `Minkovski Inequality`, we have
$$
\|\mathbf{w}\|_p+\|\mathbf{v}\|_p\geq\|\mathbf{w+v}\|_p
$$

In this set of regulizers, we have special cases of:
- $p=1$, which is Lasso. 
- $p=2$, which is Ridge.

To be detailed, compare to Ridge regulizer, Lasso regulizer penalize large $\mathbf{\overline{w}}$ less, and in cases leads to no closed-form solution. However, there are certain reasons to choose Lasso for:
- Tends to produce sparse solution (with some dimensions of the matrix =0). This helps automatic feature selection, and reduces the complexity of the models.

## Variance and Bias

The formula fits that

$$
\text{predicted error}=bias^2+Variance+noise
$$

And we conclude an empirical experience that there is `no free lunch`: ^9a1bd6
- low bias leads to high variance. This is correlated with overfitting, which would be sensitive to data changes.
- low variance leads to high bias. This is correlated with underfitting, since the solutions can't fit the data well.

A trend plot goes like this:
![[Pasted image 20250728132113.png]]


## Overfitting and Underfitting

`Overfitting`: If test set error is much higher than training set error, we say model overfits.
- Solution: make less complex model, or remove features, or add data

`Underfitting`: If training set error is high, we say model fits poorly.
- Solution: use more complex model, or add new features, or add data

Improve the generalizability of the model, we need to test it using certain test sets, which is about 20% of the whole dataset. Then further, we need to test the hyperparameters of the same model. Therefore validation set is required.

## Data Division and Cross Validation

A simple and effective procedure:
- Split the data into training dataset $\mathcal{D}_{\text{train}}$ and validation dataset $\mathcal{D}_{\text{validation}}$.
- Pick $m$ different model sets $\phi_1,\phi_2,...\phi_m$
- Train models on $\mathcal{D}_{\text{train}}$ and get one hypothesis $h:\mathcal{X}\to\mathcal{Y}$ for each $\phi$s.
- Compute the error on $\mathcal{D}_{\text{validation}}$ and choose the lowest
	$$
	h^*=\arg\min_{h\in\mathcal{H}}\operatorname{Error}_{\mathcal{D}_{\text{validation}}}(h)
	$$

Then cross-validation is introduced to highly make use of the training dataset.

`Cross Validation` goes like this:
- same as above.
- For each possible split of dataset into training set $\mathcal{D}$ and validation set $\mathcal{D}'$:$$
	\{\mathcal{D}_1,\mathcal{D}_1'\},\{\mathcal{D}_2,\mathcal{D}_2'\}
	,...,\{\mathcal{D}_K,\mathcal{D}_K'\}$$
- Compute the error of each hypothesis on validation set $D'$: $\operatorname{E}_{D'}(h_{\phi,\mathcal{D}})$.
- Estimate the error of the model as the average of error on each $D'$:$$
	\phi^*=\arg\min_{\phi_i}\frac{1}{K}\sum^K_{j=1}\operatorname{E}_{D'_j}(h_{\phi_i,\mathcal{D'}_j})$$
Then how to pick splits is questioned. Often we take two strategies:
- <b>Leave-one-out Cross Validation</b>. A brief example is given in [[SVM (Support Vector Machine)#Model Evaluation|SVM Model Evaluation]]. Briefly, in each split, the validation is just one sample. This obtains several characters as:
	- Accurate: training using nearly all training data
	- Slow: experience $n$ times of iterations.

- **$\mathbf{n}$-fold Cross Validation**. Similar to above, while this time, we first divide the data into different folds of equal number of data points. Then we conduct works on folds. Below is an example of $n=6$.

	![[Pasted image 20250728140939.png]]

## Future Reading

