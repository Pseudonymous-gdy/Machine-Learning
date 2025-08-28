
**Document Class** #DSAA2011  #Supervised-Learning #Regression 

**Table of Contents:**
- [[#Motivation for ridge regression|Motivation for ridge regression]]
- [[#Formulation: New Objective Function|Formulation: New Objective Function]]
- [[#Techniques: Dual Form|Techniques: Dual Form]]
- [[#Codes|Codes]]

**Ridge performs better for sparse data in redundant features and preventing overfitting.**

---
## Motivation for ridge regression

If given two data points, how would you conduct a linear regression? In certain cases, limited samples lead to model's overfitting given lots of attributes and few samples.

On the other hand, by least squares, we might have$$
\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\in\mathbb{R}^{d+1}
$$obtained from$$
\min\ J(\mathbf{w})=(\mathbf{Xw}-y)^T(\mathbf{Xw}-y)
$$Where $X\in\mathbb{R}^{m\times(d+1)}$. Thus, the design matrix is `very 'wide'` (in shape), unlikely to have full column rank, which might cause $\mathbf{X}^T\mathbf{X}$ uninvertible, and the model fails.

Therefore, we require one method that stabilize and robustify the solution while keeping the other process same.

## Formulation: New Objective Function

For a fixed $\lambda\geq0$, consider
$$\begin{aligned}
J(\mathbf{w})&=\sum^m_{i=1}(f_{\mathbf{w}}(\mathbf{x}_i)-y_i)^2+\lambda\sum^d_{i=0}w_j^2\\
&=(\mathbf{Xw}-\mathbf{y})^T(\mathbf{Xw}-\mathbf{y})+\lambda\mathbf{w}^T\mathbf{w}\ \text{($w_0=b$, the bias)}
\end{aligned}$$
From the formula, the term $\lambda\mathbf{w}^T\mathbf{w}$ encourages the weight vector to have small components (known as <font color=red>shrinkage</font>). This is called `ridge regression` or `Tikhonov regularization`. When $\lambda=0$, recover to usual linear regression.

Based on this, we might find a solution that minimize the new objective function. Basically,
$$\begin{aligned}
\mathbf{w}&=\arg\min_{\mathbf{w}}J(\mathbf{w})\\&=\arg\min_{\mathbf{w}}(\mathbf{Xw}-\mathbf{y})^T(\mathbf{Xw}-\mathbf{y})+\lambda\mathbf{w}^T\mathbf{w}\\
\frac{\partial J}{\partial \mathbf{w}}&=2\mathbf{X}^T\mathbf{X}\mathbf{w}-2\mathbf{X}^T\mathbf{y}+2\lambda\mathbf{w}=0\\
\mathbf{X}^T\mathbf{y}&=(\mathbf{X}^T\mathbf{X}+\lambda)\mathbf{w}\\
\mathbf{w}&=(\mathbf{X}^T\mathbf{X}+\lambda \mathbf{I}_{d+1})^{-1}\mathbf{X}^T\mathbf{y}
\end{aligned}$$
This guarantees the invertibility of $\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I}$, since
$$
\mathbf{v}^T(\mathbf{X}^T\mathbf{X}+\lambda \mathbf{I}_{d+1})\mathbf{v}=(\mathbf{Xv})^T\mathbf{Xv}+\lambda\mathbf{v}^T\mathbf{v}>0
$$
Making the matrix positive definite, and obviously invertible. This is because
$$
\dim \mathcal{N}(\mathbf{X}^T\mathbf{X}+\lambda \mathbf{I}_{d+1})=\{\omega|(\mathbf{X}^T\mathbf{X}+\lambda \mathbf{I}_{d+1})\omega=\mathbf{0}_{d+1}\}=\{\mathbf{0}_{d+1}\}
$$
Indicating the full column rank of the matrix. (Otherwise, there exists one vector $\mathbf{v}$, making $\mathbf{v}^T(\mathbf{X}^T\mathbf{X}+\lambda \mathbf{I}_{d+1})\mathbf{v}=0$, contradiction!)

## Techniques: Dual Form

How to get the inverse of a matrix of $\mathbb{R}^{(d+1)\times(d+1)}$? By `Gaussian Elimination` the time complexity is $O((d+1)^3)=O(d^3)$. Therefore, if $m\geq d$, it might be still applicable; if $m<d$, it might be better to apply <font color=red>dual form</font> to ==lower computational costs==, which is equal to the primal form:$$
(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I}_{d+1})^{-1}\mathbf{X}^T\mathbf{y}=\mathbf{X}^T(\mathbf{X}\mathbf{X}^T+\lambda\mathbf{I}_{m})^{-1}\mathbf{y}
$$To further investigate on this, we first exhibit *Woodbury formula*:$$\begin{aligned}
(\mathbf{I}+\mathbf{U}\mathbf{V})^{-1}&=\mathbf{I}-\mathbf{U}(\mathbf{I}+\mathbf{V}\mathbf{U})^{-1}\mathbf{V}\\
\\
(\mathbf{I}+\mathbf{U}\mathbf{V})\left[\mathbf{I}-\mathbf{U}(\mathbf{I}+\mathbf{V}\mathbf{U})^{-1}\mathbf{V}\right]&=\mathbf{UV}-\mathbf{U}\mathbf{V}\mathbf{U}(\mathbf{I}+\mathbf{V}\mathbf{U})^{-1}\mathbf{V}+\mathbf{I}-\mathbf{U}(\mathbf{I}+\mathbf{V}\mathbf{U})^{-1}\mathbf{V}\\
&=\mathbf{U}\left(\mathbf{I}-(\mathbf{I}+\mathbf{V}\mathbf{U})^{-1}\right)\mathbf{V}+\mathbf{I}-\mathbf{U}\mathbf{V}\mathbf{U}(\mathbf{I}+\mathbf{V}\mathbf{U})^{-1}\mathbf{V}\\
&=\mathbf{U}\left(\mathbf{I}-(\mathbf{I}+\mathbf{V}\mathbf{U})^{-1}-\mathbf{V}\mathbf{U}(\mathbf{I}+\mathbf{V}\mathbf{U})^{-1}\right)\mathbf{V}+\mathbf{I}\\
&=\mathbf{U\cdot0\cdot V}+\mathbf{I}=\mathbf{I}
\end{aligned}$$Then, we have
$$\begin{aligned}
\mathbf{X}^T(\mathbf{XX}^T+\lambda\mathbf{I}_{m})^{-1}\mathbf{y}&=\lambda^{-1}\mathbf{X}^T(\mathbf{I}_m+\lambda^{-1}\mathbf{XX}^T)^{-1}\mathbf{y}\\
&=\lambda^{-1}\mathbf{X}^T\Bigl[\mathbf{I}_{m}-\lambda^{-1}\mathbf{X}(\mathbf{I}_{d+1}+\lambda^{-1}\mathbf{X}^T\mathbf{X})\mathbf{X}^T\Bigr]\mathbf{y}\\
&=\lambda^{-1}\Bigl[\mathbf{I}_{d+1}-\lambda^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{I}_{d+1}+\lambda^{-1}\mathbf{X}^T\mathbf{X})^{-1}\Bigr]\mathbf{X}^T\mathbf{y}\\
&=\lambda^{-1}\Bigl[\mathbf{I}_{d+1}-\mathbf{X}^T\mathbf{X}(\lambda\mathbf{I}_{d+1}+\mathbf{X}^T\mathbf{X})^{-1}\Bigr]\mathbf{X}^T\mathbf{y}\\
&=\lambda^{-1}\left[(\lambda\mathbf{I}_{d+1}+\mathbf{X}^T\mathbf{X}-\mathbf{X}^T\mathbf{X})(\mathbf{I}_{d+1}+\lambda^{-1}\mathbf{X}^T\mathbf{X})^{-1}\right]\mathbf{X}^T\mathbf{y}\\
&=(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I}_{d+1})^{-1}\mathbf{X}^T\mathbf{y}
\end{aligned}$$
Through this operation, we turn the primal calculation into a simpler dual one.

## Codes

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np
X_train = np.array([[1,-7],[1,-5],[1,1],[1,5]])
y_train = np.array([[-1],[1],[1],[-1]])

clf = Ridge(alpha=1.0,solver='lsqr').fit(X_train, y_train)
y = clf.predict(np.array([[1,3],[1,5],[1,-3]]))
print(y)
```
