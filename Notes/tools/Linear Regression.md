
# Least Squares and Linear Regression

## Motivation for linear regression

Given 5 Indicators:
- Hours Studied $x_1$, Previous Scores $x_2$, Extracurricular Activities $x_3$, Sleep Hours $x_4$, Sample question papers practiced $x_5$.
- Which factor is most important for determining the student's performance $y$?

![[Pasted image 20250715161107.png]]
Computed post-hoc: $\beta^{\ast}_{\text{SCIE}}=.40$, $\beta^{\ast}_{\text{GLCM}}=.38$, $\beta^{\ast}_{\text{MATH}}=.15$, $\beta^{\ast}_{\text{JOYREAD}}=.06$, $\beta^{\ast}_{\text{METASUM}}=.04$.

`Form of Data:`
$$
\mathbf{x}_i = \begin{pmatrix}
x_{i,1}\\
x_{i,2}\\
x_{i,3}\\
x_{i,4}\\
x_{i,5}
\end{pmatrix},y_i\ for\ i\in\{1,2,...,1000\}
$$


## Linear Regression

Linear regression is a _linear_ approach for modelling the relationship between _a scalar response_ $y$ and one or more explanatory variables $\mathbf{x}$ (a vector of attribute values).

### LR with Scalar
`Target`: find $\mathbf{w}\in\mathbb{R}^d$ satisfying the linear system with `design matrix` $\mathbf{X}$ and `target vector` $\mathbf{y}$:
$$
\mathbf{Xw}=\begin{pmatrix}
x_{1,1} & x_{1,2} & ... & x_{1,d}\\
x_{2,1} & x_{2,2} & ... & x_{2,d}\\
...& & &...\\
x_{m,1} & x_{m,2} & ... & x_{m,d}
\end{pmatrix}\mathbf{w}=\mathbf{y}=\begin{pmatrix}
y_1\\y_2\\...\\y_m
\end{pmatrix}
$$

**With Offset:**
$$
f_{\mathbf{w},b}(x)=\begin{pmatrix}b\\ \mathbf{w}\end{pmatrix}^T
\begin{pmatrix}1\\ \mathbf{x}\end{pmatrix}
$$

`Error`: for $i\in{1,2,...,m}$, we define
$$\begin{aligned}
e_i&:=f_{\mathbf{w},b}(\mathbf{x}_i)-y_i\text{ (per-sample loss/objective function)}\\
\text{Loss}(\mathbf{w},b)&=\frac{1}{m}\sum_{i=1}^m(f_{\mathbf{w},b}(\mathbf{x}_i)-y_i)^2\text{ (squared loss/Objective function)}\\
&=(\mathbf{X}\mathbf{\overline{w}}-\mathbf{y})^T(\mathbf{X}\mathbf{\overline{w}}-\mathbf{y})
\end{aligned}$$
To minimize loss function, we apply derivatives:
$$\begin{aligned}
J(\mathbf{\overline{w}})&:=(\mathbf{X}\mathbf{\overline{w}}-\mathbf{y})^T(\mathbf{X}\mathbf{\overline{w}}-\mathbf{y})\\
&=\mathbf{\overline{w}}^T\mathbf{X}^T\mathbf{X\overline{w}}-2\mathbf{\overline{w}}^T(\mathbf{X}^T\mathbf{y})+\mathbf{y}^T\mathbf{y}\\
\frac{dJ}{d\mathbf{\overline{w}}}&= 2\mathbf{X}^T\mathbf{X}\mathbf{\overline{w}}-2\mathbf{X}^T\mathbf{y}\\
2\mathbf{X}^T\mathbf{X}\mathbf{\overline{w}}&=2\mathbf{X}^T\mathbf{y}\\
\mathbf{\overline{w}}&=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
\end{aligned}$$

It is a global minimum, since
$$\begin{aligned}
Method\ 1:\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 
\frac{dJ}{d\mathbf{\overline{w}}d\mathbf{\overline{w}}^T}&=2\mathbf{X}^T\mathbf{X},\ 
while\ \forall\mathbf{v}\in\mathbb{R}^d,\\
\mathbf{v}^T\frac{dJ}{d\mathbf{\overline{w}}d\mathbf{\overline{w}}^T}\mathbf{v}&=2(\mathbf{Vx})^T(\mathbf{Vx})\geq0\\
\text{Which reveals that the Hessian} &\ \text{Matrix is semi-positive definite.}\\
\text{Therefore the loss function is} &\ \text{convex, which proves the claim.}\\
Method\ 2:
\lambda J(\mathbf{\overline{w}}_1)+(1-\lambda)J(\mathbf{\overline{w}}_2)&=
\lambda(\mathbf{X\overline{w}}_1-\mathbf{y})^T(\mathbf{X\overline{w}}_1-\mathbf{y})+(1-\lambda)(\mathbf{X\overline{w}}_2-\mathbf{y})^T(\mathbf{X\overline{w}}_2-\mathbf{y})\\
&=\lambda\mathbf{\overline{w}}_1^T\mathbf{X}^T\mathbf{X\overline{w}}_1+(1-\lambda)\mathbf{\overline{w}}_2^T\mathbf{X}^T\mathbf{X\overline{w}}_2\\&\ \ \ -2((1-\lambda)\mathbf{\overline{w}}_2+\lambda\mathbf{\overline{w}}_1)^T(\mathbf{X}^T\mathbf{y})+\mathbf{y}^T\mathbf{y}\\
&=J(\lambda\mathbf{\overline{w}}_1+(1-\lambda)\mathbf{\overline{w}}_2)\\&\ \ \ -[\lambda\mathbf{\overline{w}}_1+(1-\lambda)\mathbf{\overline{w}}_2]^T\mathbf{X}^T\mathbf{X}(\lambda\mathbf{\overline{w}}_1+(1-\lambda)\mathbf{\overline{w}}_2)\\
&\ \ \ \ +\lambda\mathbf{\overline{w}}_1^T\mathbf{X}^T\mathbf{X\overline{w}}_1+(1-\lambda)\mathbf{\overline{w}}_2^T\mathbf{X}^T\mathbf{X\overline{w}}_2\\
&= \lambda(1-\lambda)\bigl[\mathbf{\overline{w}}_1^T\mathbf{X}^T\mathbf{X\overline{w}}_1+\mathbf{\overline{w}}_2^T\mathbf{X}^T\mathbf{X\overline{w}}_2\\ &\ \ \ \ \ -\mathbf{\overline{w}}_2^T\mathbf{X}^T\mathbf{X\overline{w}}_1-\mathbf{\overline{w}}_2^T\mathbf{X}^T\mathbf{X\overline{w}}_1\bigr]+J(\lambda\mathbf{\overline{w}}_1+(1-\lambda)\mathbf{\overline{w}}_2)\\
&\geq J(\lambda\mathbf{\overline{w}}_1+(1-\lambda)\mathbf{\overline{w}}_2)\\
Since&\ \bigl[\mathbf{\overline{w}}_1^T\mathbf{X}^T\mathbf{X\overline{w}}_1+\mathbf{\overline{w}}_2^T\mathbf{X}^T\mathbf{X\overline{w}}_2-\mathbf{\overline{w}}_2^T\mathbf{X}^T\mathbf{X\overline{w}}_1-\mathbf{\overline{w}}_2^T\mathbf{X}^T\mathbf{X\overline{w}}_1\bigr]\\
&=(\mathbf{X}(\mathbf{\overline{w}}_1-\mathbf{\overline{w}}_2))^T(\mathbf{X}(\mathbf{\overline{w}}_1-\mathbf{\overline{w}}_2))\geq0
\end{aligned}$$
After the loss minimization, the contributions between $y$ and each $x$ could be observed.

### LR with Vector

Here, $y$ is turned into vectors, where the target changes to
$$
\mathbf{Xw}=\begin{bmatrix}
1&x_{1,1} & x_{1,2} & ... & x_{1,d}\\
1&x_{2,1} & x_{2,2} & ... & x_{2,d}\\
...& & & &...\\
1&x_{m,1} & x_{m,2} & ... & x_{m,d}
\end{bmatrix}\begin{bmatrix}b_1&b_2&...&b_h\\
w_{1,1}&w_{1,2}&...&w_{1,h}\\
w_{2,1}&w_{2,2}&...&w_{2,h}\\
...&&&...\\
w_{d,1}&w_{d,2}&...&w_{d,h}
\end{bmatrix}=\mathbf{y}=\begin{bmatrix}
y_{1,1}&y_{1,2}&...&y_{1,h}\\
y_{2,1}&y_{2,2}&...&y_{2,h}\\
...&&&...\\
y_{m,1}&y_{m,2}&...&y_{m,h}
\end{bmatrix}
$$

The loss function is also turned into:
$$\begin{aligned}
\text{Loss}(\mathbf{w})&=\sum^h_{i=1}\sum^m_{j=1}(f_{\mathbf{w},i}(\mathbf{x}_j)-y_{j,i})^2\\
&=\sum^h_{i=1}(\mathbf{X}\mathbf{w}_i-\mathbf{y}_i)^T(\mathbf{X}\mathbf{w}_i-\mathbf{y}_i)\text{ (means the ith column vector)}
\end{aligned}$$

The results, if by taking the partial derivatives, would be:
$$\begin{aligned}
J(\mathbf{w})&:=\sum^h_{i=1}(\mathbf{X}\mathbf{w}_i-\mathbf{y}_i)^T(\mathbf{X}\mathbf{w}_i-\mathbf{y}_i)\\
0=\frac{\partial J}{\partial \mathbf{w}_i}&=\frac{\partial (\mathbf{X}\mathbf{w}_i-\mathbf{y}_i)^T(\mathbf{X}\mathbf{w}_i-\mathbf{y}_i)}{\partial \mathbf{w_i}}\\
Therefore\ \mathbf{w}_i&=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}_i\\
And\ \mathbf{w}&=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
\end{aligned}$$
Advanced techniques:
$$\begin{aligned}
J(\mathbf{w})&=tr((\mathbf{Xw}-\mathbf{y})^T(\mathbf{Xw}-\mathbf{y}))\\
&=tr(\mathbf{A}^T\mathbf{A})\\
dJ &= d\operatorname{tr}\left(tr(\mathbf{A}^T\mathbf{A})\right)\\
&= \operatorname{tr}\left(d\ \mathbf{A}^T\mathbf{A}\right)= \operatorname{tr}\left(2\mathbf{A}^Td\mathbf{A}\right)\\
&= \operatorname{tr}\left(2\mathbf{A}^T\mathbf{X}d\mathbf{w}\right)=\operatorname{tr}\left(2(\mathbf{Xw}-\mathbf{y})^T\mathbf{X}d\mathbf{w}\right)\\
\text{Since }dJ &= \operatorname{tr}\left( \left( \frac{\partial J}{\partial w} \right)^T dw \right)\text{, we have}\\
\frac{\partial J}{\partial\mathbf{w}}&=2(\mathbf{Xw}-\mathbf{y})^T\mathbf{X}=0
\end{aligned}$$
# LR vs. MLE

Think about the loss:
$$
J(\mathbf{w},b)=\frac{1}{m}\sum^m_{i=1}(f_{\mathbf{\overline{w}},b}(\mathbf{x}_i)-y_i)^2=\frac{1}{m}\sum^m_{i=1}(\mathbf{\overline{w}}^T\mathbf{x}_i+b-y_i)^2
$$
This could be somehow explained by `Maximum Likelihood Estimator`. Namely, suppose
$$
e_i:=y_i-\mathbf{\overline{w}}^T\mathbf{x}_i-b\sim\mathcal{N}(0,\sigma^2)
$$
Where it could be or approximately be when $m\to\infty$ for the distribution of the mean, which is
$$
\bar{X}_n\xrightarrow{d}\mathcal{N}(0,\sigma^2)
$$
This triggers us to further hypothesize that the single error also follows a Gaussian Distribution such that the mean of their sum is Gaussian.

Under the circumstances, the target functions is guaranteed with unbiasedness, consistency, and reaches Cramér-Rao variance lower bound (asymptotically optimal), and equals with BLUE (Best Linear Unbiased Estimator) (Gauss-Markov theorem).

Then,
$$\begin{aligned}
p(y_i|x_i;\mathbf{w},\sigma^2)&=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\Bigr(-\frac{(y_i-\mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2} \Bigl)\\
K(\mathbf{w},\sigma^2):=\log L(\mathbf{w},\sigma^2;\{\mathbf{x}_i,\mathbf{y}_i\})&=\sum^m_{i=1}\log\Bigl(\frac{1}{\sqrt{2\pi\sigma^2}}\Bigr)-\sum^m_{i=1}\frac{(y_i-\mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\text{ (like the loss function)}\\
\frac{\partial K}{\partial \mathbf{w}}&=\frac{1}{\sigma^2}\sum^m_{i=1}(y_i-\mathbf{w}^T\mathbf{x}_i)\mathbf{x}_i=0,\ then\\
\sum^m_{i=1}(\mathbf{w}^T\mathbf{x}_i)\mathbf{x}_i&=\sum^m_{i=1}y_i\mathbf{x_i},\ which\ is\\
\mathbf{X}^T\mathbf{X}\mathbf{w}&=\mathbf{X}^T\mathbf{y}\\
(Complement:)\\
\frac{\partial K}{\partial \sigma}&=-m\cdot\frac{1}{\sigma}+\frac{1}{\sigma^3}\sum^m_{i=1}(y_i-\mathbf{w}^T\mathbf{x}_i)^2=0,\ then\\
\sigma^2&=\frac{1}{m}\sum^m_{i=1}(y_i-\mathbf{w}^T\mathbf{x}_i)^2
\end{aligned}$$

# Codes

The model could be directly written into:
```python
import numpy as np
from numpy.linalg import inv
X=# a matrix
y=# a matrix/a vector
# model fit
w=inv(X.T @ X) @ X.T @ y
# model predict
Xt = # new data
y_predict = Xt @ w
```

Where you might turn it into a class:
```python
import numpy as np
from numpy.linalg import inv
class LR():
	def __init()__:
		self.w = 0
	
	def train(X, y):
		self.w = inv(X.T @ X) @ X.T @ y
		
	def predict(x):
		return x @ self.w
```


# Further Reading

[[Polynomial Regression]]
[[Ridge Regression]]
[[Linear Classification]]