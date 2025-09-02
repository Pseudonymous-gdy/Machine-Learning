
Principal Component Analysis is a `linear dimensionality reduction technique`. This method finds new axes (principal components, linear combinations of existing axes) that capture most important patterns in the data, which are:
- Maximize Variance: difference/distance between points are important information that should be kept as much as possible ^45ff9c
- Minimize reconstruction error: use lower-dimensional representations to recover the original data with minimized loss ^f8e289

In other words, we model the original vectors $x_i\sim\sum^k_{j=1}z_{ij}\mathbf{v}_j$, with $\mathbf{v}_j$ as the best directions to capture important patterns.

# Motivation: Goal and Methods

Since we've mentioned our goals as [[#^45ff9c|maximizing variance]] and [[#^f8e289|minimizing error]], we define the projection of PCA as$$z\in \mathbb{R}^{n\times d}=Xu,u\in\mathbb{R}^{p\times d},X\in\mathbb{R}^{n\times p}$$Then the Variance of projection could be parsed into$$\text{Cov}(z)=\text{Cov}(Xu)=u^T\text{Cov}(X)u=u^T\Sigma u$$Where $\Sigma=\frac{1}{n}X^TX$. Please notice that:
- here, $X$ is the matrix we hope to reduce the dimensions, with rows as observations
- From the formula we could see that the variance of $z$ is dependent on $u$, therefore in order to control the size of $u$.

For convenience, we suppose $d=1$ here, where the projection is on to a line. Then we have
$$\begin{aligned}\text{Var}(z)&=\text{Cov}(z)=u^T\Sigma u\\
\Sigma &= \frac{1}{n}X^TX\end{aligned}$$Without the loss of generalizability, we could suppose $\|u\|=1$, which we turn the problems into$$\begin{aligned}\max_uu^T\Sigma u,&\quad\text{subject to }u^Tu=1\\
\Rightarrow \max_u&\frac{u^T\Sigma u}{u^Tu}\quad\text{(equivalently)}\end{aligned}$$thus finding out about our goal in this method, which is to figure out an optimal $u$.

A "closed" form solution could be derived, with the techniques of
1. Spectral Decomposition
	For a $n\times n$ real symmetric matrix $A$, we have $A=\sum^n_{i=1}\lambda_ie_ie_i^T$, where $\lambda_i$ is $i$-th eigenvalue, and $e_i$ is the corresponding normalized eigenvector, and $A^{1/2}=\sum^n_{i=1}\sqrt{\lambda_i}e_ie_i^T$.
	(Sketch): 
		Step 1. For a real symmetric matrix A, the eigenvectors of different eigenvalues are orthogonal (check $e_i^TAe_j$ and $e_j^TAe_i$)
		Step 2. There is an orthogonal matrix $Q$ such that $AQ=Q\Lambda,Q^{-1}=Q^T$
		Step 3. We then have $A=Q\Lambda Q^T$, and $A^{1/2}=Q\Lambda^\frac{1}{2}Q^T$
2. Maximization of Quadratic Forms for Points on the Unit Sphere
	Let $B$ be a positive semi-definite matrix with eigenvalues $\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_n\geq0$ and associated normalized eigenvectors $e_1,\cdots,e_n$, and $u$ is a unit vector, then$$\begin{aligned}\max _{u \neq 0} u^{\top} B u&=\lambda_{1}\\
	\min _{u \neq 0} u^{\top} B u&=\lambda_{n}\\
	\max _{u \neq 0, u \perp e_{1}, ..., e_{k}} u^{\top} B u&=\lambda_{k+1}\end{aligned}$$
	(Sketch):
		Step 1. $u^TBu=u^TB^{1/2}B^{1/2}u=u^TQ\Lambda^{1/2}Q^TQ\Lambda^{1/2}Q^Tu=(Q^Tu)^T\Lambda(Q^Tu)$
		Step 2. We have $v:=Q^Tu$ still a unit vector, and $v^T\Lambda v=\sum^n_{i=1}\lambda_iv_i^2\leq\lambda_1$. The equivalence could hold.
		Step 3. We have $u=Qv=\sum^n_{i=1}v_ie_i\Rightarrow u^Te_1=v_1=0$. Similarly, we have $v_2=\cdots=v_k=0$, then $v^T\Lambda v=\sum^n_{i=k+1}\lambda_iv_i^2\leq\lambda_{k+1}$.
Therefore, what we need to do is to figure out the greatest eigenvalues and corresponding projection matrix $u=\frac{1}{\sqrt{k}}\sum^k_{i=1}e_i$.

As the general case, for $\Sigma\in\mathbb{R}^{p\times p}=\frac{1}{n}X^TX=U\Lambda U^T,U\in\mathbb{R}^{p\times p},\Lambda=\begin{pmatrix}\lambda_1&0&\cdots&0\\0&\lambda_2&\cdots&0\\\cdots&\cdots&\cdots&\cdots\\0&0&\cdots&\lambda_p\end{pmatrix}$, we would select columns of $U_k\in\mathbb{R}^{p\times k}$ corresponding to $k$ largest eigenvalues. Then the projection of new data in lower dimensions would be:$$Z=XU_k,Z\in\mathbb{R}^{n\times k}$$
With the `proportions of variance explained` as $\sum^k_{i=1}\lambda_i/\sum^p_{j=1}\lambda_j$. This leads to the methodology of our algorithms.

# Algorithms

`Steps`:
1. Input: $X \in \mathbb{R}^{n ×p}$ (n samples, p features).
2. Standardize the Data: Scale features to zero mean and unit variance:  $X_{k} \leftarrow(X_{k}-\mu_{k}) / \sigma_{k}$ or $X' \leftarrow(V^{1 / 2})^{-1}(X-\mu)$
3. Compute Covariance Matrix: Calculate $\Sigma =\frac{1}{n} X^{\prime \top} X'$
4. Eigenvalue Decomposition: Find eigenvectors (principal components) and eigenvalues (variance) of $\Sigma$
5. Select Principal Components: Choose top $k$ eigenvectors based on highest eigenvalues (e.g., 95% variance).
6. Project Data: Transform data to lower-dimensional space: $Z=X' W$ , where w is the matrix of selected eigenvectors.

# Hyperparameter: k

- For visualization, use k=2 or 3
- For other applications, use elbow method in scree plot, which plots eigenvalues against the corresponding Principal Component number. Basically, the eigenvalues drop quickly before the elbow but not after the elbow.

# Reconstruction on Error Minimization

Given data $X\in\mathbb{R}^{n\times p}$ (n samples, p features), and assume that we have the projection matrix $U\in\mathbb{R}^{p\times k}$. $U$ contains $k$ orthonormal bases that define the subspace. Then the projected points are $Z=XU$, with the reconstructed points as $\hat{X}=ZU^T=XUU^T$.

By Pythagorean theorem, the reconstruction error is parsed into:$$\underbrace{\sum^n_{i=1}\|x_i-\hat{x}_i\|^2}_\text{Reconstruction Error}=\underbrace{\sum^n_{i=1}\|x_i\|^2}_\text{Const}-\underbrace{\sum^n_{i=1}\|\hat{x}_i\|^2}_\text{Variance}$$And we find that the reconstruction error is minimized once variance is maximized.

# Cautions

1. We need Standardizations!
	If not scaling each coordinate appropriately, the coordinate with the igher variance will dominate the principal component, then the PCA results will be sensitive to the units of each coordinate.
2. We `can` use $\Sigma=\frac{1}{n-1}X^TX$
	It is always alright, since the eigenvalues and the eigenvectors are the same.
3. Difference between linear regression and PCA:
	- In PCA, all dimensions are treated equally, while label column is special in linear regression
	- In PCA, we minimize `the sum of the squared perpendicular distances` after projecting X into the subspace spanned by Principal Components. In linear regression, we minimize the sum of squared errors after projecting y into the subspace spanned by X (not perpendicular).
4. We might use Kernel PCA! [[#Kernel PCA]]

# Pros & Cons

1. Pros
	- Linear method, simpler to implement
	- Computationally efficient for large datasets
	- Reduces multi-collinearity effectively
2. Cons
	- Assumes linear relationships in data
	- Less effective for preserving local structure
	- May lose critical information if variance is not representative
	- Sensitive to outliers (variants: [[#Robust PCA]])


# Kernel PCA




# Robust PCA



