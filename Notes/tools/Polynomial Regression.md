# Motivation

When the data points are not approximately linear, affine functions may not do the great job to show the contributions of independent variables. Also, to be specific, while solving XOR problems like:
$$
\mathbf{x}_i=[\pm 1,\pm 1],y_i=\pm1
$$
To be detailed,
$$\begin{cases}
\mathbf{x}_1=[+1,+1]^T,y_1=1\\
\mathbf{x}_2=[-1,+1]^T,y_2=-1\\
\mathbf{x}_3=[+1,-1]^T,y_3=-1\\
\mathbf{x}_4=[-1,-1]^T,y_1=1\\
\end{cases}$$
This can't be resolved using linear classifier but quadratic function, i.e. $f(x_1,x_2)=x_1x_2$.
This leads to the discussion of polynomial regression, which is an extension of linear regression.

# Polynomial Regression

Take the cubic model, for example, we form the model as
$$
f_{\mathbf{w}}(x_1,x_2,...,x_d)=w_0+\sum^d_{i=1}w_ix_i+\sum_{1\leq i\leq j\leq d}w_{i,j}x_ix_j+\sum_{1\leq i\leq j\leq k\leq d}w_{i,j,k}x_ix_jx_k
$$

In total, there are $C_{d-1}^0+C_d^1+C_{d+1}^2+C_{d+2}^3=C_{d+3}^3$ items in the formula. If extend to the degree of $p$ for the polynomial, we have the number of items as $C_{d+p}^p$, which is a very large number when $d$ and/or $p$ is large.

Therefore, the model is parsed as
$$\begin{aligned}
f_{\mathbf{w}}(\mathbf{x})&=\mathbf{Pw}\\
&=\begin{bmatrix}
1&x_{1,1}&...&x_{1,d}&...&x_{1,i}x_{1,j}&...&x_{1,i}x_{1,j}x_{1,k}&...\\
1&x_{2,1}&...&x_{2,d}&...&x_{2,i}x_{2,j}&...&x_{2,i}x_{2,j}x_{2,k}&...\\
...&&&&&&&&...\\
...&&&&&&&&...\\
...&&&&&&&&...\\
1&x_{m,1}&...&x_{m,d}&...&x_{m,i}x_{m,j}&...&x_{m,i}x_{m,j}x_{m,k}&...\\
\end{bmatrix}\cdot\begin{bmatrix}
w_0\\ w_1\\ w_d\\ ...\\ w_{i,j}\\ ...\\ w_{i,j,k}\\ ...
\end{bmatrix}\ \ \ \ \ \text{ ($w_0=b$)}\\
&=\begin{bmatrix}y_1\\ y_2\\ ...\\ y_m\end{bmatrix}=\mathbf{y}
\end{aligned}$$

Where $\mathbf{P}=\begin{bmatrix}-\mathbf{p}_1^T-\\-\mathbf{p}_2^T-\\ ... \\-\mathbf{p}_m^T-\\\end{bmatrix}\in\mathbb{R}^{m\times C_{d+p}^p}$, and similarly, we have
$$
\mathbf{w}=(\mathbf{P}^T\mathbf{P})^{-1}\mathbf{P}^T\mathbf{y}
$$
`Dual Form`:
$$
\mathbf{w}=\mathbf{P}^T(\mathbf{P}\mathbf{P}^T)^{-1}\mathbf{y}
$$
For detailed discussion, see [[Ridge Regression#Dual Form]]. It is obvious that the two forms are equal, while under the requisite that the two matrix $\mathbf{P}\mathbf{P}^T,\mathbf{P}^T\mathbf{P}$ are all invertible. This further indicates that $P$ is a full-ranked square matrix.


# Usage

For `regression applications`:
- learn continuous-valued $y$ by using either primal or dual forms
- Prediction:
	$$
	\hat{y}_{new}=\mathbf{p}_{new}^T\mathbf{w}
	$$
	The essence of such operation is to create more features based on the existing attributes. The features are taken into account in the model with the attributes.

For `classification applications`:
- learn discrete-valued $y\in\{-1,+1\}$ (for binary classification) or one-hot encoded $\mathbf{Y}$ (for multi-class classification with number of classes $C$) using either primal or dual forms
- Binary Prediction:
	$$
	\hat{y}_{new}=\text{sign}(\mathbf{p}_{new}^T\mathbf{w})
	$$
- Multi-class Prediction:
	$$
	\hat{y}_{new} = \arg\max_{k\in\{1,2,...,C\}}(\mathbf{p}_{new}^T\mathbf{W}[k])
	$$


# Codes

An example solution of XOR problem is introduced, where we apply polynomial regression.

```python
import numpy as np
from numpy.linalg import inv
from numpy.linalg import matrix_rank
from sklearn.preprocessing import PolynomialFeatures
X = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
y = np.array([[1],[-1],[-1],[1]])
order = 2
poly = PolynomialFeatures(order)
P = poly.fit_transform(X)

PY = np.vstack((P.T,y.T))
print(matrix_rank(P)<matrix_rank(PY)) # check the invertibility through column rank

# dual solution m < d (without ridge)
w_dual = P.T @ inv(P @ P.T) @ y
print(w_dual)
```
